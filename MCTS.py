from __future__ import annotations

"""
Generic, parallel-safe MCTS backbone.

This module is intentionally opinionated about *structural* concurrency, and
intentionally unopinionated about domain semantics.

Highlights
==========

1. Minimal base node type
   The base node stores only:
     * a parent pointer
     * a mapping from edge labels to child nodes

   Any domain-specific state belongs in subclasses.

2. Thread-safe tree mutation
   The backbone protects node structure with per-node locks and statistics with
   separate per-node locks. Downstream code must never mutate ``node.children``
   or ``node.parent`` directly after a node has entered the tree.

3. Expansion batches instead of single-child expansion
   ``MCTSProblem.expand(...)`` may return a *forest* of proposed insertions as a
   topologically ordered flat list of tuples ``(anchor, edge, child,
   should_rollout)``.

   ``should_rollout`` means exactly this:
     * ``True``  -> the newly attached child should be queued for rollout now
     * ``False`` -> materialize the child now, but do not queue rollout now

   A child marked ``False`` can still be selected later, expanded later, and
   receive value from descendants later.

4. Conservative conflict handling
   The backbone attaches proposed edges one at a time. If an edge already
   exists, the existing child stays canonical. The incoming child is considered
   to have lost that publication race.

   The backbone deliberately does *not* try to reinterpret or transplant the
   losing child's descendants onto the canonical winner. Any later batch items
   anchored at the losing child object are skipped. This keeps semantic
   reconciliation in user code instead of baking it into the generic backbone.

5. Asynchronous rollout work
   Expansion may enqueue zero, one, or many rollout targets. Each pending
   rollout contributes virtual visits along its ancestor path to the root.
   When the rollout completes, those virtual visits are resolved into ordinary
   MCTS statistics during backpropagation.

6. Pluggable worker scheduler
   ``MCTSWorkerScheduler`` manages how expansion jobs and rollout jobs are
   executed. The default scheduler uses a shared pool of worker threads in two
   phases:
     * phase 1: root-selection / expansion jobs
     * phase 2: rollout jobs produced by phase 1

   More elaborate schedulers can be supplied by users.

Expansion batch contract
========================

The list returned by ``expand(...)`` must satisfy all of the following:

* It is topologically ordered.
* Each tuple is ``(anchor, edge, child, should_rollout)``.
* ``anchor`` is either:
    - already in the live MCTS tree before ``expand`` started, or
    - a ``child`` that appeared earlier in the same returned list.
* The proposed insertions form a forest.
* No proposed child has two different parents within one batch.
* The same ``(anchor, edge)`` pair does not appear twice within one batch.

Processing rules:
* The backbone processes the returned list in order.
* Each parent->edge->child attachment is published independently.
* If an attachment conflicts with an existing edge, the incoming child loses by
  default, and any later items anchored at that losing child object are skipped.
* A successful attachment with ``should_rollout=True`` creates one pending
  rollout task.

Algorithmic note
================

This module strictly generalizes classical parallel MCTS. Classical behavior is
recovered when all of the following hold:

* expansion always returns a list of length 0 or 1,
* each successful expansion schedules at most one rollout,
* anchored expansion entry points are not used, and
* the scheduler is configured in the usual one-job-per-worker style.
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Optional, Protocol, Sequence, TypeVar, runtime_checkable
import math
import random
import threading
from weakref import WeakKeyDictionary


EdgeT = TypeVar("EdgeT")
NodeT = TypeVar("NodeT", bound="MCTSNode[Any]")

# A single expansion item:
#   anchor          -> existing tree node to attach under
#   edge            -> edge label under that anchor
#   child           -> proposed new child object
#   should_rollout  -> whether the newly attached child should be queued now
ExpansionItem = tuple[NodeT, EdgeT, NodeT, bool]


class MCTSNode(Generic[EdgeT]):
    """
    Minimal base node.

    Only tree structure lives here. Domain-specific fields should be added in a
    subclass.

    ``__weakref__`` is present because the MCTS engine stores per-node locks and
    statistics in ``WeakKeyDictionary`` side tables keyed by node identity.
    """

    __slots__ = ("parent", "children", "__weakref__")

    def __init__(self, parent: Optional["MCTSNode[EdgeT]"] = None) -> None:
        self.parent: Optional["MCTSNode[EdgeT]"] = parent
        self.children: dict[EdgeT, "MCTSNode[EdgeT]"] = {}


@dataclass(frozen=True)
class NodeStats:
    """
    Read-only statistics snapshot for one node.

    ``in_flight`` counts virtual visits contributed by rollout tasks that have
    been queued but have not finished yet.
    """

    visits: int
    value_sum: float
    in_flight: int

    @property
    def effective_visits(self) -> int:
        return self.visits + self.in_flight

    @property
    def value_mean(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass(frozen=True)
class RolloutTask(Generic[NodeT]):
    """
    A single pending rollout.

    ``path`` is stored root->...->target. Virtual visits are charged to this
    exact path when the task is queued, and resolved on the same path when the
    rollout completes or fails.
    """

    target: NodeT
    path: tuple[NodeT, ...]


@runtime_checkable
class MCTSProblem(Protocol[NodeT, EdgeT]):
    """
    Domain interface for the MCTS backbone.

    Reentrancy requirements
    -----------------------
    All protocol methods must be safe to call concurrently from multiple
    threads. In particular, ``expand`` and ``rollout`` must not rely on hidden
    shared mutable state unless that state is protected externally by the user.

    Tree-mutation rule
    ------------------
    ``expand`` must *not* mutate ``node.children`` or ``node.parent`` directly.
    It must only *describe* proposed insertions by returning a batch. The
    backbone is the sole owner of live tree publication.
    """

    def is_terminal(self, node: NodeT) -> bool:
        """Return True exactly when ``node`` should be treated as terminal."""
        ...

    def expand(self, node: NodeT, mcts: "MCTS[NodeT, EdgeT]") -> list[ExpansionItem[NodeT, EdgeT]]:
        """
        Describe zero or more proposed insertions under ``node`` and/or its
        descendants.

        The returned list must satisfy the batch contract described in this
        module docstring.
        """
        ...

    def rollout(self, node: NodeT, rng: random.Random) -> float:
        """
        Evaluate ``node`` and return a scalar reward.

        ``node`` may be terminal or non-terminal; that choice is domain-specific.
        A common pattern is to mark terminal nodes in domain state and simply let
        ``rollout`` short-circuit when it sees such a node.
        """
        ...

    def backprop_transform(self, node: NodeT, reward: float, depth_from_leaf: int) -> float:
        """
        Transform reward during backpropagation.

        Default usage:
          * single-agent search         -> identity
          * alternating zero-sum search -> often sign flip on odd depths
        """
        return reward


ScoreFn = Callable[["MCTS[NodeT, EdgeT]", NodeT, EdgeT, NodeT], float]


class MCTSWorkerScheduler(Protocol[NodeT, EdgeT]):
    """
    Scheduler interface for expansion and rollout work.

    ``anchors`` is a sequence of expansion jobs.

    Job semantics:
      * ``None``      -> perform one ordinary root-originating job:
                         select from the root until a node is expandable or
                         terminal, then publish one expansion batch or queue one
                         terminal rollout.
      * ``node``      -> perform one anchored job at that exact node, without a
                         fresh selection pass from the root.

    A scheduler may use any worker topology it likes as long as these semantic
    effects occur.
    """

    def run(
        self,
        mcts: "MCTS[NodeT, EdgeT]",
        problem: MCTSProblem[NodeT, EdgeT],
        anchors: Sequence[Optional[NodeT]],
    ) -> None:
        ...


class _Stats:
    """Mutable per-node statistics record protected by its own lock."""

    __slots__ = ("mu", "visits", "value_sum", "in_flight")

    def __init__(self) -> None:
        self.mu = threading.Lock()
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.in_flight: int = 0


class MCTS(Generic[NodeT, EdgeT]):
    """
    Generic, parallel-safe MCTS engine.

    Structural ownership
    --------------------
    Once a node has entered the live tree, only this class is allowed to modify
    its ``parent`` pointer and ``children`` mapping. Downstream code may inspect
    tree structure via snapshots, but must not mutate it.

    Two public ways to generate work
    --------------------------------
    * ``run(...)`` / ``run_parallel(...)`` use ordinary root-originating jobs.
    * ``run_at(...)`` / ``run_parallel_at(...)`` target specific anchor nodes.

    Anchored jobs are the escape hatch for applications that want to cache
    domain-specific semantic information, reconcile it after a publication race,
    and then explicitly continue expansion from the canonical live node that won
    the race.
    """

    def __init__(
        self,
        root: NodeT,
        *,
        exploration_constant: float = math.sqrt(2.0),
        score_fn: Optional[ScoreFn[NodeT, EdgeT]] = None,
        seed: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        self.root: NodeT = root
        self.c: float = exploration_constant
        self.debug: bool = debug

        # Global lock protecting get-or-create access to side tables.
        self._tables_mu = threading.Lock()

        # Per-node structure lock: guards publication of parent/children links.
        self._struct_locks: "WeakKeyDictionary[NodeT, threading.RLock]" = WeakKeyDictionary()

        # Per-node statistics record.
        self._stats: "WeakKeyDictionary[NodeT, _Stats]" = WeakKeyDictionary()

        # Selection scoring policy. Signature is always (mcts, parent, edge, child).
        self.score_fn: ScoreFn[NodeT, EdgeT] = score_fn or (lambda m, p, e, c: m._ucb1_score(p, e, c))

        # Thread-local RNGs avoid contention and accidental cross-thread coupling.
        self._rng_local = threading.local()
        self._base_seed = seed if seed is not None else random.randrange(1 << 63)

    # ------------------------------------------------------------------
    # Side-table accessors
    # ------------------------------------------------------------------

    def _struct_mu(self, node: NodeT) -> threading.RLock:
        with self._tables_mu:
            mu = self._struct_locks.get(node)
            if mu is None:
                mu = threading.RLock()
                self._struct_locks[node] = mu
            return mu

    def _get_stats(self, node: NodeT) -> _Stats:
        with self._tables_mu:
            st = self._stats.get(node)
            if st is None:
                st = _Stats()
                self._stats[node] = st
            return st

    def _thread_rng(self) -> random.Random:
        rng = getattr(self._rng_local, "rng", None)
        if rng is None:
            tid = threading.get_ident()
            self._rng_local.rng = random.Random(self._base_seed ^ (tid * 0x9E3779B97F4A7C15))
            rng = self._rng_local.rng
        return rng

    # ------------------------------------------------------------------
    # Public stats API
    # ------------------------------------------------------------------

    def stats_snapshot(self, node: NodeT) -> NodeStats:
        st = self._get_stats(node)
        with st.mu:
            return NodeStats(visits=st.visits, value_sum=st.value_sum, in_flight=st.in_flight)

    def visits(self, node: NodeT, *, effective: bool = False) -> int:
        s = self.stats_snapshot(node)
        return s.effective_visits if effective else s.visits

    def value_sum(self, node: NodeT) -> float:
        return self.stats_snapshot(node).value_sum

    def value_mean(self, node: NodeT) -> float:
        return self.stats_snapshot(node).value_mean

    # ------------------------------------------------------------------
    # Tree snapshots
    # ------------------------------------------------------------------

    def children_items_snapshot(self, node: NodeT) -> list[tuple[EdgeT, NodeT]]:
        """
        Snapshot ``node.children.items()`` under the node's structure lock.

        Downstream code should always use this helper instead of iterating the
        live dict directly under concurrency.
        """
        with self._struct_mu(node):
            return list(node.children.items())

    def children_keys_snapshot(self, node: NodeT) -> set[EdgeT]:
        """Snapshot the current set of expanded edge labels under ``node``."""
        with self._struct_mu(node):
            return set(node.children.keys())

    # ------------------------------------------------------------------
    # Conflict handling and publication helpers
    # ------------------------------------------------------------------

    def conflict_resolver(
        self,
        anchor: NodeT,
        edge: EdgeT,
        incoming_child: NodeT,
        existing_child: NodeT,
    ) -> None:
        """
        Hook called when ``anchor.children[edge]`` already exists.

        Default behavior is intentionally conservative: keep the existing child
        canonical and do nothing else.

        Overriding guidance:
          * Treat this as a *local* hook. It is the right place to cache or copy
            domain-specific information elsewhere, but not the right place to
            rewrite the tree.
          * Do not mutate ``anchor.children`` or ``existing_child.parent`` here.
          * Do not perform heavy blocking work here; the expansion batch that hit
            the conflict can always be resumed later through anchored jobs.
        """
        return None

    def try_attach_child(self, anchor: NodeT, edge: EdgeT, child: NodeT) -> tuple[NodeT, bool]:
        """
        Try to attach ``child`` under ``anchor`` via ``edge``.

        Returns ``(canonical_child, created)``.

        ``created`` is True exactly when ``child`` became the live child under
        ``anchor.children[edge]``.

        Publication order matters:
          1. set ``child.parent``
          2. publish ``anchor.children[edge] = child``

        This prevents readers from observing a reachable child whose parent
        pointer is still unset.
        """
        existing: Optional[NodeT] = None
        with self._struct_mu(anchor):
            existing = anchor.children.get(edge)
            if existing is None:
                child.parent = anchor
                anchor.children[edge] = child
                if self.debug:
                    assert anchor.children[edge] is child
                    assert child.parent is anchor
                return child, True

        # Conflict handling stays outside the structure lock so that user hooks
        # do not block unrelated structural publication.
        assert existing is not None
        self.conflict_resolver(anchor, edge, child, existing)
        return existing, False

    def create_child(self, anchor: NodeT, edge: EdgeT, child: NodeT) -> NodeT:
        """Ensure there is a child under ``(anchor, edge)`` and return the canonical node."""
        canonical, _created = self.try_attach_child(anchor, edge, child)
        return canonical

    # ------------------------------------------------------------------
    # Selection policy
    # ------------------------------------------------------------------

    def _ucb1_score(self, parent: NodeT, edge: EdgeT, child: NodeT) -> float:
        """
        Vanilla UCB1 score with virtual visits.

        ``value_mean`` uses backed-up values only.
        ``effective_visits = visits + in_flight`` is used for the exploration term
        so that pending rollout work discourages redundant contention.
        """
        parent_s = self.stats_snapshot(parent)
        child_s = self.stats_snapshot(child)

        n_child = child_s.effective_visits
        if n_child == 0:
            return float("inf")

        n_parent = max(1, parent_s.effective_visits)
        exploit = child_s.value_mean
        explore = self.c * math.sqrt(math.log(n_parent) / n_child)
        return exploit + explore

    def select_child(self, node: NodeT) -> tuple[EdgeT, NodeT]:
        """
        Pick the best child under the current selection policy.

        A snapshot is taken first so scoring never iterates a live dict that may
        be mutated concurrently by another thread.
        """
        items = self.children_items_snapshot(node)
        if not items:
            raise ValueError("select_child called on a node with no children")

        best_score = -float("inf")
        best: list[tuple[EdgeT, NodeT]] = []
        rng = self._thread_rng()

        for edge, child in items:
            score = self.score_fn(self, node, edge, child)
            if score > best_score:
                best_score = score
                best = [(edge, child)]
            elif score == best_score:
                best.append((edge, child))

        return rng.choice(best)

    # ------------------------------------------------------------------
    # Path helpers and virtual visits
    # ------------------------------------------------------------------

    def path_to_root(self, node: NodeT) -> tuple[NodeT, ...]:
        """
        Return the unique parent chain from the root to ``node``.

        Parent pointers are write-once for live tree nodes, so reading the chain
        does not require holding structure locks.
        """
        rev: list[NodeT] = []
        cur: Optional[NodeT] = node
        while cur is not None:
            rev.append(cur)
            cur = cur.parent  # type: ignore[assignment]
        rev.reverse()
        return tuple(rev)

    def _add_virtual_visits(self, path: Sequence[NodeT]) -> None:
        for node in path:
            st = self._get_stats(node)
            with st.mu:
                st.in_flight += 1

    def _resolve_virtual_visits_only(self, path: Sequence[NodeT]) -> None:
        for node in reversed(path):
            st = self._get_stats(node)
            with st.mu:
                st.in_flight -= 1
                if self.debug:
                    assert st.in_flight >= 0, "in_flight went negative"

    def _make_rollout_task(self, node: NodeT) -> RolloutTask[NodeT]:
        path = self.path_to_root(node)
        self._add_virtual_visits(path)
        return RolloutTask(target=node, path=path)

    # ------------------------------------------------------------------
    # Expansion-batch publication
    # ------------------------------------------------------------------

    def _publish_expansion_batch(
        self,
        batch: Sequence[ExpansionItem[NodeT, EdgeT]],
    ) -> list[RolloutTask[NodeT]]:
        """
        Publish one expansion batch and return the rollout tasks it generated.

        Processing rule on conflict:
        if a proposed child loses its publication race, any later items in the
        same batch anchored at that losing child object are skipped.

        This is a purely structural rule. The backbone does not try to decide
        whether descendants of the losing child are semantically reusable under
        the canonical winner. Applications that want to salvage more information
        can do so explicitly by caching that information and later issuing an
        anchored job at the canonical node.
        """
        rollout_tasks: list[RolloutTask[NodeT]] = []
        dead_anchor_ids: set[int] = set()

        if self.debug:
            seen_pairs: set[tuple[int, Any]] = set()
            seen_children: set[int] = set()

        for anchor, edge, child, should_rollout in batch:
            if id(anchor) in dead_anchor_ids:
                continue

            if self.debug:
                pair = (id(anchor), edge)
                assert pair not in seen_pairs, "duplicate (anchor, edge) inside one expansion batch"
                seen_pairs.add(pair)
                assert id(child) not in seen_children, "one proposed child appears under two parents in one batch"
                seen_children.add(id(child))

            canonical, created = self.try_attach_child(anchor, edge, child)
            if not created or canonical is not child:
                dead_anchor_ids.add(id(child))
                continue

            if should_rollout:
                rollout_tasks.append(self._make_rollout_task(child))

        return rollout_tasks

    # ------------------------------------------------------------------
    # Expansion jobs
    # ------------------------------------------------------------------

    def _run_root_job(self, problem: MCTSProblem[NodeT, EdgeT]) -> list[RolloutTask[NodeT]]:
        """
        One ordinary root-originating job.

        Tree policy:
          * if the current node is terminal, queue one rollout for it
          * else call ``expand`` on the current node
          * if ``expand`` returns a non-empty batch, publish it and stop
          * otherwise descend to the best child and continue
          * a non-terminal dead-end with no children and an empty batch is a no-op

        This keeps the engine generic: the backbone never needs a separate
        notion of "fully expanded". That decision is entirely encoded in the
        domain's ``expand`` implementation.
        """
        node = self.root
        while True:
            if problem.is_terminal(node):
                return [self._make_rollout_task(node)]

            batch = problem.expand(node, self)
            if batch:
                return self._publish_expansion_batch(batch)

            items = self.children_items_snapshot(node)
            if not items:
                return []

            _edge, node = self.select_child(node)

    def _run_anchored_job(self, problem: MCTSProblem[NodeT, EdgeT], anchor: NodeT) -> list[RolloutTask[NodeT]]:
        """
        One anchored job at exactly ``anchor``.

        No fresh selection pass from the root is performed. This is the public
        escape hatch for applications that want to continue from a canonicalized
        node after reconciling domain-specific semantics externally.
        """
        if problem.is_terminal(anchor):
            return [self._make_rollout_task(anchor)]

        batch = problem.expand(anchor, self)
        if not batch:
            return []
        return self._publish_expansion_batch(batch)

    def _dispatch_job(
        self,
        problem: MCTSProblem[NodeT, EdgeT],
        anchor: Optional[NodeT],
    ) -> list[RolloutTask[NodeT]]:
        return self._run_root_job(problem) if anchor is None else self._run_anchored_job(problem, anchor)

    # ------------------------------------------------------------------
    # Rollout completion and backpropagation
    # ------------------------------------------------------------------

    def _backprop_path(
        self,
        problem: MCTSProblem[NodeT, EdgeT],
        path: Sequence[NodeT],
        reward: float,
    ) -> None:
        for depth_from_leaf, node in enumerate(reversed(path)):
            r = float(problem.backprop_transform(node, reward, depth_from_leaf))
            st = self._get_stats(node)
            with st.mu:
                st.in_flight -= 1
                st.visits += 1
                st.value_sum += r
                if self.debug:
                    assert st.in_flight >= 0, "in_flight went negative"

    def _execute_rollout_task(self, problem: MCTSProblem[NodeT, EdgeT], task: RolloutTask[NodeT]) -> None:
        """
        Execute one pending rollout task.

        If rollout raises an exception, only the virtual visits are rolled back;
        no backed-up value is recorded for that failed task.
        """
        try:
            reward = problem.rollout(task.target, self._thread_rng())
        except BaseException:
            self._resolve_virtual_visits_only(task.path)
            raise
        self._backprop_path(problem, task.path, reward)

    # ------------------------------------------------------------------
    # Public execution API: sequential
    # ------------------------------------------------------------------

    def run(self, problem: MCTSProblem[NodeT, EdgeT], *, iterations: int) -> None:
        """
        Sequential root-originating execution.

        Each root job is run immediately, and all rollout tasks generated by that
        job are executed immediately afterward on the same thread.
        """
        for _ in range(iterations):
            tasks = self._run_root_job(problem)
            for task in tasks:
                self._execute_rollout_task(problem, task)

    def run_at(self, problem: MCTSProblem[NodeT, EdgeT], anchors: Iterable[NodeT]) -> None:
        """
        Sequential anchored execution.

        Each supplied anchor gets exactly one anchored job.
        """
        for anchor in anchors:
            tasks = self._run_anchored_job(problem, anchor)
            for task in tasks:
                self._execute_rollout_task(problem, task)

    # ------------------------------------------------------------------
    # Public execution API: scheduler-driven
    # ------------------------------------------------------------------

    def run_parallel(
        self,
        problem: MCTSProblem[NodeT, EdgeT],
        *,
        iterations: int,
        scheduler: MCTSWorkerScheduler[NodeT, EdgeT],
    ) -> None:
        """
        Run ``iterations`` ordinary root-originating jobs through ``scheduler``.
        """
        scheduler.run(self, problem, [None] * iterations)

    def run_parallel_at(
        self,
        problem: MCTSProblem[NodeT, EdgeT],
        anchors: Sequence[NodeT],
        *,
        scheduler: MCTSWorkerScheduler[NodeT, EdgeT],
    ) -> None:
        """
        Run one anchored job for each node in ``anchors`` through ``scheduler``.
        """
        scheduler.run(self, problem, list(anchors))

    # ------------------------------------------------------------------
    # Action recommendation
    # ------------------------------------------------------------------

    def best_edge(self, node: Optional[NodeT] = None, *, by: str = "visits") -> EdgeT:
        """
        Recommend a move from ``node`` (default: root).

        ``by="visits"`` uses real visits only, not effective visits, because the
        usual interpretation is "which move has accumulated the strongest backed
        up evidence so far" rather than "which move is currently attracting the
        most pending work".
        """
        node = self.root if node is None else node
        items = self.children_items_snapshot(node)
        if not items:
            raise ValueError("No children available to choose from")
        if by not in ("visits", "value"):
            raise ValueError('by must be "visits" or "value"')

        best_metric = -float("inf")
        best_edges: list[EdgeT] = []
        rng = self._thread_rng()

        for edge, child in items:
            stats = self.stats_snapshot(child)
            metric = stats.visits if by == "visits" else stats.value_mean
            if metric > best_metric:
                best_metric = metric
                best_edges = [edge]
            elif metric == best_metric:
                best_edges.append(edge)

        return rng.choice(best_edges)


class PhasedThreadPoolScheduler(Generic[NodeT, EdgeT]):
    """
    Default scheduler using one shared worker pool and two phases.

    Phase 1: expansion jobs
      Workers pull ordinary root jobs and/or anchored jobs until the expansion
      queue is empty. Each completed expansion job may append zero, one, or many
      rollout tasks to a shared rollout queue.

    Phase 2: rollout jobs
      After all workers finish phase 1, a barrier releases them into phase 2,
      where they drain the rollout queue.

    This scheduler is simple and predictable rather than maximally aggressive.
    More overlap between expansion and rollout can be implemented in custom
    schedulers if desired.
    """

    def __init__(self, *, workers: int) -> None:
        if workers <= 0:
            raise ValueError("workers must be positive")
        self.workers = workers

    def run(
        self,
        mcts: MCTS[NodeT, EdgeT],
        problem: MCTSProblem[NodeT, EdgeT],
        anchors: Sequence[Optional[NodeT]],
    ) -> None:
        if not anchors:
            return

        root_job = object()
        expand_jobs: list[object] = [root_job if anchor is None else anchor for anchor in anchors]
        rollout_jobs: list[RolloutTask[NodeT]] = []

        expand_mu = threading.Lock()
        rollout_mu = threading.Lock()
        barrier = threading.Barrier(self.workers)

        first_exc: list[BaseException] = []
        exc_mu = threading.Lock()

        def record_exception(exc: BaseException) -> None:
            with exc_mu:
                if not first_exc:
                    first_exc.append(exc)
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass

        def pop_expand_job() -> tuple[bool, Optional[NodeT]]:
            with expand_mu:
                if not expand_jobs:
                    return False, None
                job = expand_jobs.pop()
            if job is root_job:
                return True, None
            return True, job  # type: ignore[return-value]

        def pop_rollout_job() -> Optional[RolloutTask[NodeT]]:
            with rollout_mu:
                if not rollout_jobs:
                    return None
                return rollout_jobs.pop()

        def push_rollout_tasks(tasks: Sequence[RolloutTask[NodeT]]) -> None:
            if not tasks:
                return
            with rollout_mu:
                rollout_jobs.extend(tasks)

        def worker_main() -> None:
            try:
                # Phase 1: drain expansion jobs.
                while True:
                    has_job, anchor = pop_expand_job()
                    if not has_job:
                        break
                    tasks = mcts._dispatch_job(problem, anchor)
                    push_rollout_tasks(tasks)

                barrier.wait()

                # Phase 2: drain rollout jobs.
                while True:
                    task = pop_rollout_job()
                    if task is None:
                        break
                    mcts._execute_rollout_task(problem, task)

            except threading.BrokenBarrierError:
                # Another worker reported a real exception and aborted the round.
                return
            except BaseException as exc:
                record_exception(exc)
                return

        threads = [threading.Thread(target=worker_main, name=f"mcts-worker-{i}") for i in range(self.workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if first_exc:
            raise RuntimeError("PhasedThreadPoolScheduler worker failed") from first_exc[0]

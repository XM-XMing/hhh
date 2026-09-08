# Flight Planning Context

This context defines the evaluation and guarded off-policy learning language for
the Unity flight-planning system. The terms below separate environment results
from protocol evidence and from policy-selection evidence.

## Execution and Evaluation

**Completed Primitive Execution**:
A schema-v3 primitive whose every submitted frame is applied on consecutive
physics ticks and whose endpoint is the post-integration state of its last frame.
*Avoid*: successful command, full receipt

**Terminal Abort**:
An incomplete primitive execution stopped by an explicit environment-terminal
state; it preserves the real applied-frame prefix and is an episode outcome, not
a completed primitive.
*Avoid*: partial completion, protocol failure

**Protocol Error**:
A primitive receipt or execution failure that cannot be explained by a valid
environment-terminal state and therefore invalidates the affected run.
*Avoid*: terminal abort, soft failure

**Formal Evaluator**:
The fixed-mission, deterministic-argmax Unity evaluation path governed by the
schema-v3 execution and terminal-abort contracts.
*Avoid*: ad hoc rollout, training collection

## Guarded Off-Policy Learning

**Fresh Diagnostic Run**:
An AWAC run initialized only from the frozen BC checkpoint, with a new replay
and output lineage, used to identify bottlenecks rather than to tune or select a
deployable policy.
*Avoid*: resume, continuation, historical comparison run

**Resolved Run Contract**:
The immutable artifact containing the runner command, resolved CLI arguments,
relevant environment, and runtime identities actually used to start a fresh
diagnostic run.
*Avoid*: source default, historical configuration

**Diagnostic Horizon**:
The fixed replay-step budget of a fresh diagnostic run, selected to expose the
Critic, Actor, candidate-interaction funnel, and AWAC weighting without tuning
them; the current horizon is 18,000 environment steps.
*Avoid*: training-to-convergence budget, hyperparameter sweep

**Aborted Pre-Learning Run**:
A run that ended before its first learner update; its replay and checkpoints
are preserved only as diagnostic evidence and cannot seed a Fresh Diagnostic
Run or support a learning-performance conclusion.
*Avoid*: resumable partial run, warm-start replay

**Terminal-Abort Runtime Verification**:
The first live confirmation that a classifier-validated Terminal Abort becomes
one normal collision transition without interrupting training; it verifies the
training/environment contract rather than policy performance.
*Avoid*: collision retry, learning result

**Candidate Interaction Funnel**:
The ordered evidence path from a candidate action being evaluated through its
eligibility and scheduled opportunity to its execution in replay.
*Avoid*: candidate rate

**Candidate Eligibility**:
The conjunction of finite candidate values, BC-top-k support, action difference
from the anchor, twin-Critic ordering, and positive conservative advantage.
*Avoid*: candidate opportunity, candidate execution

**Candidate Opportunity**:
A transition selected by the deterministic committed-prefix quota scheduler; an
eligible candidate at an opportunity must execute.
*Avoid*: eligibility, post-eligibility rejection

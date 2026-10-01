/**
 * Domain types for the buoy telemetry recovery problem.
 *
 * Each observed packet carries:
 *  - a unique caller-assigned id (download order is meaningless)
 *  - a counter remainder modulo `modulus` (the rotation/wrap counter)
 *  - an integer closed time interval [timeLower, timeUpper] during which
 *    the packet was actually transmitted
 *
 * The service jointly assigns every packet:
 *  - a distinct absolute counter, congruent to its remainder modulo modulus,
 *    inside the absolute-count search window
 *  - an integer transmit timestamp inside its closed interval
 * such that for every pair of adjacent observed packets (in recovered order),
 * the observed time difference lies inside the sampling window implied by
 * their counter difference.
 */

export interface PacketInput {
  /** Caller-assigned unique packet identifier (string or number). */
  id: string | number;
  /** Observed counter remainder r, required: 0 <= r < modulus. */
  remainder: number;
  /** Inclusive integer lower bound of the transmit time interval. */
  timeLower: number;
  /** Inclusive integer upper bound of the transmit time interval. */
  timeUpper: number;
}

export interface SolveRequest {
  packets: PacketInput[];
  /** Counter modulus M (rotation period), integer >= 2. */
  modulus: number;
  /** Inclusive absolute-counter search window lower bound. */
  countLower: number;
  /** Inclusive absolute-counter search window upper bound. */
  countUpper: number;
  /** Minimum interval (inclusive) between adjacent samples. */
  minInterval: number;
  /** Maximum interval (inclusive) between adjacent samples. */
  maxInterval: number;
  /**
   * Optional nominal sampling interval. When present, totalJitterBudget must
   * be present too and minInterval <= nominalInterval <= maxInterval.
   * Enables the cumulative-jitter constraint: for every adjacent observed pair
   * with selected time gap dt and counter gap d, |dt - d * nominalInterval| is
   * the pair jitter, and the sum of all pair jitters must not exceed the
   * budget. The constraint is solved jointly with order/counts/timestamps.
   */
  nominalInterval?: number;
  /** Non-negative inclusive cap on the sum of all per-pair jitters. */
  totalJitterBudget?: number;
}

/** Enabled cumulative-jitter constraint (both optional request fields given). */
export interface JitterConfig {
  nominalInterval: number;
  totalJitterBudget: number;
}

/** Per-adjacent-pair constraint check evidence. */
export interface AdjacencyEvidence {
  /** Position in the recovered order, 0-based (evidence[i] joins slot i -> i+1). */
  index: number;
  fromId: string | number;
  toId: string | number;
  fromCount: number;
  toCount: number;
  /** toCount - fromCount (always >= 1; observed packets are distinct). */
  countGap: number;
  fromTime: number;
  toTime: number;
  /** toTime - fromTime (always positive for a consistent solution). */
  timeGap: number;
  /** Inclusive feasible time-difference range for this counter gap. */
  allowedTimeGap: { min: number; max: number };
  /**
   * Nominal time gap d * nominalInterval; present only when the
   * nominalInterval/totalJitterBudget pair is enabled.
   */
  nominalTimeGap?: number;
  /**
   * |timeGap - nominalTimeGap| for this pair; present only when the jitter
   * budget is enabled.
   */
  jitter?: number;
  /**
   * Sum of jitter over adjacency entries 0..index (inclusive); present only
   * when the jitter budget is enabled.
   */
  cumulativeJitter?: number;
  /** Number of unobserved absolute counters strictly between the pair. */
  missingBetween: number;
  /** Congruence note for the destination packet. */
  congruence: { remainder: number; modulus: number };
  /** toCount mod modulus equals the destination packet's remainder. */
  absoluteCountCongruent: boolean;
  /** Closed-interval containment for the chosen timestamps. */
  timeWithinInterval: {
    from: { lower: number; upper: number };
    to: { lower: number; upper: number };
  };
  satisfied: boolean;
}

export interface MissingSegment {
  /** Inclusive first missing absolute counter. */
  fromCount: number;
  /** Inclusive last missing absolute counter. */
  toCount: number;
  /** Number of counters in the segment (toCount - fromCount + 1). */
  length: number;
}

export interface AssignedPacket {
  /** 0-based position in the recovered transmission order. */
  position: number;
  id: string | number;
  absoluteCount: number;
  time: number;
  remainder: number;
  timeInterval: { lower: number; upper: number };
}

export interface SolveResult {
  order: (string | number)[];
  assignments: AssignedPacket[];
  /** Absolute counters inside [firstCount, lastCount] not assigned to a packet. */
  missingSegments: MissingSegment[];
  missingCountTotal: number;
  adjacency: AdjacencyEvidence[];
  /** Counts of the first/last observed packets. */
  observedCountRange: { first: number; last: number };
  /**
   * Jitter budget accounting; present only when nominalInterval and
   * totalJitterBudget were supplied. `used` is the sum of the per-pair
   * jitters of the recovered interpretation (always <= budget).
   */
  jitterBudget?: {
    nominalInterval: number;
    budget: number;
    used: number;
    remaining: number;
    exhausted: boolean;
  };
}

/** Stable business error codes. */
export type SolveErrorCode =
  | 'INVALID_REQUEST'
  | 'NO_CONSISTENT_INTERPRETATION';

export interface ConstraintFailureEvidence {
  /** Recovery stage at which extension failed. */
  stage: 'seed' | 'extension';
  /** Number of packets already fixed in the partial order (0 for seed failure). */
  partialLength: number;
  /** Packet ids already fixed, in partial order. */
  partialOrder: (string | number)[];
  /** The packet that could not be appended / could not be seeded. */
  candidateId: string | number;
  /**
   * Human-readable, stable description of the first constraint that could
   * not be extended.
   */
  reason: string;
  /**
   * Numeric detail:
   *  - for 'extension' adjacency failures: the count gap that was rejected,
   *  - absent when no admissible candidate count exists at all.
   */
  detail?: {
    /** Which constraint class prevents the extension. */
    cause: 'TIME_GAP' | 'COUNT_WINDOW' | 'CONGRUENCE' | 'JITTER_BUDGET';
    /** Smallest congruent counter gap considered. */
    minimalCongruentGap?: number;
    countGap?: number;
    requiredTimeGap?: { min: number; max: number };
    actualTimeGapRange?: { min: number; max: number };
    /** Residual gap range allowed by the absolute-count search window. */
    countGapWindow?: { min: number; max: number };
    /**
     * Cumulative-jitter detail, present for cause 'JITTER_BUDGET' (and
     * whenever the jitter budget is enabled). `used` is the jitter already
     * consumed by the fixed prefix; `minimumAdditional` is the smallest
     * non-negative additional jitter this extension would add (0 when a
     * jitter-free extension exists but later completion still overflows);
     * `budget` is the totalJitterBudget cap.
     */
    jitter?: {
      nominalInterval: number;
      budget: number;
      used: number;
      minimumAdditional: number;
    };
  };
}

export class SolveError extends Error {
  readonly code: SolveErrorCode;
  readonly evidence?: ConstraintFailureEvidence;

  constructor(code: SolveErrorCode, message: string, evidence?: ConstraintFailureEvidence) {
    super(message);
    this.name = 'SolveError';
    this.code = code;
    this.evidence = evidence;
  }
}

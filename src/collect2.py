from residency_headroom.workloads import Workload, WorkloadSequence

def sub(w, lo, hi, n):
    return Workload(n, w.regime, tuple(WorkloadSequence(s.source_sequence_id, i, s.segment_index,
        s.segment_label, s.domain) for i, s in enumerate(w.sequences[lo:hi])))


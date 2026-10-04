---
title: Retrieval Quality-Efficiency Pareto Frontiers Under GPU Constraints
date: 2026-10-04
status: 후보
field: LLM Agent Long-Term Memory: Empirical Efficiency Under Compute Constraints
tags: [long-term memory, LLM agents, memory management, retrieval-augmented generation, context window, memory efficiency]
source: cosci-lite 20261004-015957-llm-agent-long-term-memory-rtx-a / H4
novelty: N3
elo: 1232
---

# Retrieval Quality-Efficiency Pareto Frontiers Under GPU Constraints

## 한 줄 요약
Direct measurement of retrieval method (BM25 vs. dense vs. hybrid) and parameter (top-k, reranking) trade-offs on agent accuracy and GPU efficiency reveals which retrieval strategy dominates under each memory-latency budget.

## 문제와 동기
P7은 retrieval이 에이전트 메모리의 주요 병목임을 보였으나, 제약 환경(48GB VRAM, 저지연시간 요구)에서 어느 검색 메서드가 정확도-효율성 측면에서 우수한지 미해결. 기존 연구는 무제한 API/VRAM을 가정하므로 현실적 배포 가이드 부재. 이 연구는 검색 메서드별 정확도-VRAM-지연시간 파레토 경계를 정량화하여 실무 선택지 제공.

## 핵심 아이디어
H1의 oracle 진단(P8 방법론) 대신, 검색 품질을 직접 제어하는 실험으로 피벗. (1) 3개 검색 메서드(BM25, dense retrieval with faiss, hybrid re-ranking) 구현. (2) agent task (P1 benchmark의 3-4개 representative task)에 대해 각 메서드의 top-k ∈ {1,3,5,10}을 변동시키며 agent 성공률 측정. (3) 각 configuration의 peak VRAM, end-to-end latency, agent accuracy 기록. (4) 각 메서드별 Pareto 경계 도출 (정확도-VRAM, 정확도-지연시간). (5) task 특성(지식 베이스 크기, 쿼리 복잡도, conflict 비율)으로 최적 메서드 예측 가능성 검증. 이 접근은 oracle 구성의 계산 비용 제거, 실무적 직접성 확보, P7의 결론을 제약 환경으로 구체화.

**가설**: 48GB GPU 예산 하에서 검색 메서드(BM25, 밀집 벡터, 하이브리드) 및 top-k/재순위 파라미터를 체계적으로 변동시켰을 때, 각 에이전트 태스크 클래스에서 정확도-VRAM, 정확도-지연시간 파레토 경계가 method 간 >30% 차이를 보이며, 최적 전략이 task 특성(쿼리 복잡도, 지식 베이스 크기)에 예측 가능하게 의존할 것으로 가설.

**반증 조건**: 만약 (1) 세 메서드의 정확도-VRAM 파레토 경계가 <10% 범위 내 겹치거나 (2) 최적 메서드가 task 특성과 무관하게 일관성 있게 동일하다면, 검색 메서드 선택은 GPU 제약 환경에서 실무적 의미가 없음.

## 관련 연구
- [Fixture: Retrieval dominates agent memory bottlenecks](https://arxiv.org/abs/9926.01003), 2026 — H4 constructs explicit Pareto frontiers (accuracy-VRAM, accuracy-latency) for method triplets (BM25, dense, hybrid) under GPU constraints and maps task characteristics (knowledge base size, query complexity) to optimal method selection. P7's fixture describes only that 'retrieval quality dominates accuracy,' without clearly operationalizing efficiency metrics (VRAM, latency) under GPU constraints or task-conditional method optimality.
- [Fixture: Compression helps weak readers more than strong readers](https://arxiv.org/abs/9926.01002), 2026 — H4 focuses on retrieval method efficiency trade-offs under GPU budget; P9 focuses on how reader model strength affects post-retrieval compression gains. Orthogonal contributions.
- [Fixture: Oracle substitution diagnosis of write and retrieval loss](https://arxiv.org/abs/9926.01004), 2026 — H4 explicitly rejects oracle substitution diagnosis (P8) as computationally expensive (36+ hours oracle runs) and proposes direct measurement instead (42 hours total for 108 agent runs). This is methodological distinction: measurement approach under resource constraint vs. diagnostic approach without cost operationalization.
- [Fixture: Long-term memory benchmark for conversational agents](https://arxiv.org/abs/9925.01000), 2025 — 가설 생성 시 참고
- [Fixture: Experience replay memory for LLM agents in web tasks](https://arxiv.org/abs/9924.01007), 2024 — 가설 생성 시 참고

Novelty 판정: N3 — H4 claims to perform empirical quantification of Pareto frontiers (accuracy vs. VRAM/latency) for three retrieval methods under 48GB GPU constraints, mapping task characteristics to optimal method selection. P7's fixture indicates it established retrieval quality as a dominant bottleneck through factorial analysis, but does not explicitly claim: (1) efficiency metric operationalization (VRAM, latency) under hardware constraints, (2) Pareto frontier construction, or (3) task-conditioned method optimality. H4's specific contribution—systematic method comparison (BM25, dense FAISS, hybrid reranking) with dual efficiency objectives under quantified GPU budget—is not attested in P7's description. The hypothesis is internally consistent and feasible (HIGH feasibility rating), but faces critical implementation ambiguities identified in review (baseline fairness across asymmetric methods, embedding model specification, re-ranker choice), which suggest it requires methodological hardening before execution. Given absence of evidence in fixture descriptions that P7 performed GPU-constrained Pareto analysis with task-dependent method selection, H4 qualifies as potentially novel, but the outcome (that retrieval methods exhibit efficiency trade-offs) is somewhat predictable from P7's finding that retrieval quality dominates."

## 예상 기여
1. GPU 제약 환경에서 검색 메서드(BM25, dense, hybrid)의 정확도-효율성 파레토 경계 정량화
2. Task 특성에 따른 최적 검색 메서드 선택 휴리스틱 제시 (배포 시 즉시 활용 가능)
3. Oracle 진단 대신 직접 측정으로 계산 비용 80% 절감하는 경량 평가 방법론 제시

## 검증 계획
- 데이터셋: P1 benchmark (conversational long-term memory tasks, representative 3-4 tasks selection), Task-specific knowledge bases (varying sizes: 5K, 20K, 50K+ facts), Agent task suite (web navigation, dialogue, planning; controlled conflict/ambiguity variants)
- 베이스라인: BM25 baseline (standard sparse retrieval), Dense retrieval baseline (e.g., sentence-transformers + FAISS), Hybrid re-ranking baseline (BM25 + dense re-ranking, P3-style freshness combining), Compute-matched baseline (equal token budget, equal VRAM peak budget across methods)
- 평가 지표: Agent task success rate (accuracy on P1 benchmark per method/top-k), Peak VRAM utilization per configuration, End-to-end latency (query to agent action), Retrieval quality proxy (top-1 recall, MRR, NDCG@k), Efficiency score: accuracy / max(VRAM, latency / reference_latency), Pareto dominance count (how many baselines each method strictly dominates)
- 최소 실험 (1~2일 안에 끝나는 sanity check): 이틀 실험: (1) 첫날: BM25, dense (FAISS), 하이브리드 재순위 검색 3개 메서드 구현. P1 benchmark에서 3개 representative agent task 선정. 각 메서드 × top-k ∈ {1,3,5,10} = 12개 configuration에 대해 agent 실행 (seed 고정, n_runs=3). 각 실행의 정확도, peak VRAM, 지연시간 기록. (2) 둘째날: 3개 메서드별 파레토 경계 도출 (정확도-VRAM, 정확도-지연시간). 최적 메서드가 task 특성(지식 베이스 크기, 쿼리 복잡도)과 상관성 검증. 결론: 어느 메서드가 어느 조건에서 효율적인가.
- 필요 자원: RTX A5000 2장, ~36시간 실행 시간 (12 configurations × 3 tasks × 3 seeds = 108 agent runs, 각 ~20분). 검색 구현 준비 ~4시간. 데이터 분석/그래프 ~2시간. 총 ~42시간 (margin 포함). API 비용: 미미 (로컬 검색만 사용, 별도 LLM 호출 없음). 재현 가능성을 위해 모든 코드 공개, seed 고정, 정확한 retrieval corpus 제공.

## 리스크와 예상 반론
- Dense retrieval (FAISS)이 GPU 메모리를 과다 소비하여 BM25와 공정한 비교 불가. 대응: FAISS를 CPU에서 실행하거나, 인덱스 크기 감소(quantization, subsampling); 또는 메서드별 개별 GPU 예산 설정 후 효율성 점수 조정.
- Task 수가 적으면 (3개) 최적 메서드의 일반화 불확실. 대응: P1에서 5-6개 task로 확대; 또는 task 특성 (지식 베이스 크기, 쿼리 유형)을 명시적으로 조절하는 synthetic task 추가.
- 지연시간 측정이 불안정 (OS 스케줄링, 메모리 경합). 대응: warm-up run 후 측정, 반복 실행으로 중간값 보고, 95% 신뢰 구간 제시.
- Hybrid 메서드의 재순위 모델 선택이 결과를 지배할 수 있음. 대응: 재순위 모델을 명시적 파라미터로 취급; 여러 재순위 모델 (P3, cross-encoder) 시도하고 robust 결론 도출.
- [CRITICAL] Baseline fairness undefined: 48GB GPU budget allocation across BM25 (minimal VRAM), dense FAISS (high VRAM), and hybrid (both) is not operationalized. BM25 index costs ~0 VRAM; FAISS embedding can saturate GPU memory. How is 'equal budget' enforced without disadvantaging inherently asymmetric methods? Running dense on CPU (mentioned as risk mitigation) breaks latency fairness. Requires precise per-method VRAM/latency envelope specification before experiment.
- [CRITICAL] Monotonicity assumption unvalidated: Hypothesis assumes retrieval quality (MRR, top-1 accuracy) monotonically improves agent task success. No mechanism to verify whether the agent actually *uses* or *can exploit* higher-ranked retrieved facts. If agent reasoning is weak or top-1 suffices, further ranking improvements may not transfer to task success. Must include explicit traces showing agent utilization of retrieved facts across methods.
- [MAJOR] Dense retrieval embedding model unspecified: Hypothesis does not fix which sentence-transformer variant, model size, or training data. Choice of embedding model (e.g., small vs. large, domain-specific vs. general) can dominate method performance and VRAM. A smaller embedding model might outperform BM25 on efficiency, but this is embedding choice confound, not retrieval method superiority. Must either fix embedding model or ablate embedding size effects separately.
- [MAJOR] Re-ranker method for hybrid undefined: Hybrid re-ranking's performance is re-ranker-dependent (P3 suggests deterministic freshness; cross-encoder and LLM-based re-rankers are alternatives). Hypothesis does not specify which approach, making 'hybrid' ill-defined. Results may reflect re-ranker choice, not retrieval method trade-offs. Recommend pre-specifying re-ranker (e.g., fixed cross-encoder) or comparing multiple re-rankers as separate methods.
- [MAJOR] Task selection process invites bias: 'Representative 3-4 tasks' is post-hoc criterion. If tasks are chosen after observing preliminary results, overfitting is likely. Recommend: (a) pre-specify task selection rule (e.g., random sample from P1, or all P1 tasks), or (b) pre-register task set before seeing results. With only 3 tasks, generalization claim is weak.
- [MAJOR] Statistical power insufficient: n_runs=3 with fixed seeds is very low for agent task success (typically sparse, high-variance signal). Standard errors will be large; true Pareto differences <30% may not be detectable. Recommend n_runs ≥ 5-10, or pre-compute power analysis and report 95% confidence intervals on Pareto boundaries, not point estimates.
- [MAJOR] Noisy latency in Pareto comparisons: Latency is subject to OS scheduling, thermal throttling, and memory contention. Pareto dominance curves built from noisy latency can overlap (CIs intersect) even if methods differ in accuracy-VRAM. Hypothesis should either (1) report confidence bands on Pareto frontiers, (2) warm-up runs before measurement, or (3) filter latency-ambiguous configurations out of comparison.

## 다음 단계
- [ ] Step 1: BM25, dense FAISS, 하이브리드 재순위 3개 메서드의 프로토타입 구현 및 단일 task에서 smoke test
- [ ] Step 2: P1 benchmark에서 3개 representative agent task 선정 (지식 베이스 크기, 쿼리 복잡도 다양화)
- [ ] Step 3: 2 × 2 factorial (3 methods × 4 top-k values) × 3 tasks × 3 random seeds = 108 실험 run script 작성
- [ ] Step 4: 실행 및 peak VRAM, latency, agent success rate 로깅
- [ ] Step 5: 메서드별 파레토 경계 시각화 (accuracy-VRAM, accuracy-latency scatter plots with Pareto hull)
- [ ] Step 6: Task 특성(지식 베이스 크기, conflict 비율)과 최적 메서드 간 상관성 분석
- [ ] Step 7: 배포 휴리스틱 작성 (e.g., '지식 베이스 <10K면 BM25, >50K면 하이브리드')

## 변경 기록
- 2026-10-04: 최초 저장 (cosci-lite 자동 저장, Elo 1232, 2승 0패, novelty N3)

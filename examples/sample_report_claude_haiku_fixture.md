# LLM Agent Long-Term Memory: Empirical Efficiency Under Compute Constraints — 연구 주제 탐색 리포트

- 실행: `20261004-015957-llm-agent-long-term-memory-rtx-a` · 생성 2026-10-04T01:59:57+09:00
- 백엔드: claude=haiku
- LLM 호출: {'claude': 28} · 추정 비용(claude 보고값): $2.75
- 가설 5개 (활성 5) · 경기 6개 · 논문 12편
- 순서 바꿔 재판정한 경기 5개 중 판정 일치 100% (낮으면 심판 모델의 position bias/불안정성이 큼 → 무승부 처리됨)
- 완료 단계: parse, literature, generate:0, review:0, tournament:0, feedback:0, evolve:1, review:1, tournament:1, overview

## 연구 목표

> LLM agent의 long-term memory에서 학부생이 RTX A5000 2장으로 3~6개월 안에 할 수 있는 정량 실험 기반 연구 주제를 찾고 싶다. API 예산은 월 50달러 이하.

**핵심 질문**: 다양한 long-term memory 관리 전략에서, 제한된 계산 자원(RTX A5000 2장) 하에서 에이전트 성능과 메모리 효율성의 trade-off는 어떻게 나타나는가?

**제약/가정**
- Compute: RTX A5000 2장 (~48GB VRAM)
- Time: 3-6개월
- API budget: ≤$50/month
- Empirical/quantitative 실험 필수
- 학부생 수준의 구현 난이도
- 재현 가능한 설정

## 요약 (Research overview)

연구는 제한된 GPU 자원(RTX A5000 2장, 48GB) 하에서 LLM 에이전트의 long-term memory 관리 전략의 성능-효율성 trade-off를 실증적으로 특성화하는 목표로 진행 중입니다. 현재 상태: (1) P7이 무제약 환경에서 검색(retrieval)이 주요 병목임을 입증했으나, 극도의 VRAM 제약 환경에서의 일반화는 미검증. (2) 메모리 쓰기 측면(경험 재생, conflict resolution, 압축)의 기법들이 개별적으로 연구되었으나, 통합 수명주기 설계와 write-retrieval 상호작용은 미측정. (3) 5개 가설이 생성되었으나, 높은 Elo의 가설들(H4, H2)도 baseline 공정성, 핵심 가정의 검증 부재, 계산 비용 리스크 등의 CRITICAL 문제를 보유. 

가장 강력한 기회: (A) 검색 메서드별 Pareto 경계 정량화(H4)—P7 검색 주도성을 GPU 제약 환경에서 실증적 검증하고 비용 효율적 설정 가이드 생성(N3, 확정적 기여). (B) 가벼운 병목 진단(H1-v2)—극도의 샘플링 + 분석 경계로 제약 환경의 검색 주도성을 2-3일 내 확인(N2, 비용 효율적 전략 enabler). (C) 적응형 모델 스케일링(H2)—P11의 소형 모델 효율성을 극도의 제약에서 동적 최적화로 확장(N2, 실무 영향력 높음). 

통합 수명주기(H3)는 retrieval 주도성이 검증되지 않은 상태로 write 측 최적화에 투자하는 것은 전략적 리스크. 기존 oracle substitution(H1)은 방법론적 신규성 낮음(N1)이고 계산 비용 150+시간 위험.

핵심 불확실성: (1) 검색 vs 쓰기 손실의 상대적 크기 및 상호작용. (2) write-side 메커니즘이 domain-general한가. (3) 극도의 리소스 제약에서 새로운 병목(메모리 용량, 압축 비용)이 emergence하는가. Novelty 평가: H4(N3 확정적), H2(N2 중간), H1-v2(N2 중간), H1(N1 낮음), H3(N2 이론적 근거 약함).

| 방향 | 이유 | 관련 가설 | 첫 실험 |
|---|---|---|---|
| GPU 제약 환경에서 검색 메서드별 효율성-정확도 Pareto 경계 정량화 | P7은 무제한 API/VRAM 하에서 검색이 주요 병목임을 증명했으나, 48GB VRAM 극도의 제약에서 BM25(메모리 최소), 밀집 벡터(VRAM 고소비), 하이브리드(두 방식 조합)의 cost-benefit profile은 미측정. 각 메서드의 정확도-VRAM-지연시간 Paret… | H4 | 2일 feasibility study: (1) 첫날: 3가지 검색 메서드(BM25, FAISS 밀집 벡터, 하이브리드 재순위) 구현. P1 benchmark에서 3개 대표 agent task 선정(지식 베이스 크기, 쿼리 복잡도 다양성 기준). 각 … |
| 제약 환경에서 검색 주도성의 경량 진단 | H4의 전체 Pareto 특성화는 높은 신뢰도를 제공하지만 ~42시간의 GPU 투자 필요. H1-v2는 oracle substitution(P8)을 고분산 샘플(n=15-25)과 분석적 경계(P5, P6으로부터)로 축소하여, 2-3일 내에 검색 vs 쓰기 손실의 상대적 크기를 신뢰도 … | H1-v2 | 2-3일 실험: (1) 1일 오전: P1 benchmark에서 5개 대표 task 선정(검색 품질 표준편차, conflict 빈도 다양성 기준). 각 task마다 3개 oracle configuration: perfect write + imperfe… |
| 극도의 제약 환경에서 메모리 쓰기 모델 크기의 runtime 적응형 선택 | P11은 4B 추출 모델의 경쟁력을 보였으나, 48GB VRAM 극도의 제약에서 1.5B 소형 모델이 대부분 사용 사례에 충분하고, 메모리 volatility 증가 순간에만 더 큰 모델(7B)이 필요할 가능성 있음. Runtime metrics(memory churn rate, VRA… | H2 | 2-3일 실험: (1) 첫날: 1.5B, 2B, 4B, 7B 추출 모델을 고정 크기로 각각 구현 및 실행(P1 benchmark 대표 task, F1, VRAM, latency 측정). 각 모델의 성능 특성화. (2) 둘째날: synthetic me… |
| 메모리 충돌 유형 분류 및 도메인 간 해결 전략 효과 비교 | P3, P5는 memory conflict의 문제를 식별했으나, 충돌 유형의 체계적 분류(같은 slot vs 다른 slot, fact vs opinion vs hedged, recency vs correctness 간 conflict) 부재. 각 충돌 유형이 에이전트 성능에 미치는 차… |  | 3-4일 실험: (1) 첫날: conflict taxonomy 정의 및 synthetic conflict 데이터셋 생성(3-5개 conflict type × 3-5개 복잡도 × 2-3개 도메인 = ~30-50개 controlled scenario).… |

**핵심 불확실성**
- 검색 주도성(P7)이 극도의 GPU 제약(48GB) 환경에서 지속되는가, 아니면 메모리 용량, 압축 비용, eviction 정책이 새로운 병목으로 emergence하는가?
- 메모리 쓰기 손실(P5 간섭, P6 시간 정보 손실)과 검색 손실의 상대적 크기는 얼마인가? write-retrieval 상호작용이 oracle substitution 진단의 정확성을 얼마나 손상시키는가?
- 소형 모델(1.5B)이 극도의 VRAM 제약에서 충분한 쓰기 정확도를 제공하여 retrieval 병목을 회피할 수 있는가, 아니면 모델 크기가 추출 정확도의 hard constraint인가?
- 메모리 충돌의 유형(same-slot vs different-slot, fact vs opinion conflict), 심각도, 도메인 의존성은 무엇인가? 체계적 taxonomy가 실무 최적화를 가능하게 하는가?
- oracle substitution 진단이 n=15-25 극소 샘플에서도 신뢰할 수 있는 bottleneck attribution을 제공하는가? 분석적 경계(P5, P6)가 실제 GPU 제약 환경에서 보수적(conservative)인가, 아니면 biased인가?
- 경험 재생(P4, 웹 네비게이션), 결정론적 conflict resolution(P3), 점진적 압축(P9)의 통합이 실제로 25-35% 효율성 개선을 달성하는가, 아니면 각 메커니즘의 독립 성능 합과 동등한가?
- 다양한 도메인(웹 네비게이션, 대화형 추론, 코드 생성, 계획 수립)에서 메모리 관리 전략의 효과 크기(effect size)가 comparable한가, 아니면 도메인 특화적인가?

**추천 다음 행동**: H1-v2(경량 병목 진단)으로 시작하여 2-3일 내에 제약 환경에서 retrieval vs write 손실의 상대적 기여도를 신뢰도 있게 추정하십시오. 이 진단의 결과가 후속 연구 방향을 결정합니다: (1) 만약 retrieval loss >60%로 명확한 병목이면, H4(검색 메서드별 Pareto 경계)에 투자하여 비용 효율적 설정 가이드 생성. (2) 만약 write-loss와 retrieval-loss가 comparable하면, H2(적응형 모델 스케일링)와 H4를 병렬로 진행하여 write-retrieval 상호작용 효과 측정. (3) 만약 write-loss가 우세하거나 oracle 진단의 신뢰도가 낮으면(analytical bounds 편차 >25%), H1 전체 oracle substitution 또는 conflict taxonomy 분석으로 전환. 우선 H1-v2의 prerequisite(baseline fairness 명확화, analytical bounds 검증)을 해결한 후 실험을 시작하십시오. 성공 시 2-3일 후 명확한 연구 전략을 재정립할 수 있습니다.

## 순위 (Elo 토너먼트)

| 순위 | ID | 제목 | Elo | 승-패-무 | Novelty | Feasibility | 출처 |
|---|---|---|---|---|---|---|---|
| 1 | H4 | Retrieval Quality-Efficiency Pareto Frontiers Under GPU Constraints | 1232 | 2-0-0 | N3 | HIGH | evolved(out_of_box) ← H1 |
| 2 | H2 | Runtime-Adaptive Memory Writer Model Scaling Under Extreme Constraints | 1216 | 2-1-0 | N2 | HIGH | generated(adaptive) |
| 3 | H1-v2 | 경량 진단으로 제약 환경에서 검색 병목 검증 | 1199 | 1-1-0 | N2 | HIGH | evolved(feasibility) ← H1 |
| 4 | H1 | Efficient Oracle Substitution for Memory Bottleneck Decomposition | 1199 | 1-1-0 | N1 | HIGH | generated(measurement) |
| 5 | H3 | Integrated Memory Lifecycle: Coordinated Replay, Conflict Resolution,… | 1153 | 0-3-0 | N2 | MEDIUM | generated(cross_subfield_combination) |

> Elo는 LLM 심판의 상대 평가이며 정답(ground truth)이 아닙니다. Novelty는 검색된 논문 범위 안에서의 판단입니다.

## 상위 5개 가설 상세

### H4. Retrieval Quality-Efficiency Pareto Frontiers Under GPU Constraints

*Elo 1232 · 2승 0패 0무 · Novelty N3 · Feasibility HIGH · 전략 out_of_box*

**가설**: 48GB GPU 예산 하에서 검색 메서드(BM25, 밀집 벡터, 하이브리드) 및 top-k/재순위 파라미터를 체계적으로 변동시켰을 때, 각 에이전트 태스크 클래스에서 정확도-VRAM, 정확도-지연시간 파레토 경계가 method 간 >30% 차이를 보이며, 최적 전략이 task 특성(쿼리 복잡도, 지식 베이스 크기)에 예측 가능하게 의존할 것으로 가설.

**근거**: P7은 검색 품질이 에이전트 메모리 병목임을 보였으나, 제약 환경에서 어떤 검색 메서드가 효율적인지, 파라미터 설정이 어떻게 trade-off에 영향을 미치는지 미상. H1의 oracle 진단은 (1) 계산 비용이 높고 (2) P5, P6의 write-retrieval 상호작용을 무시하며 (3) 효율성 정규화가 임의적임. 대신, 검색 품질을 직접 변동시키고 agent 성과(정확도, VRAM, 지연시간)를 측정하면, 실무 배포를 위한 직접적 가이드를 생성. P9는 압축이 약한 reader에 더 도움이 됨을 보였으므로, 검색-reader 상호작용도 측정 대상.

**반증 조건**: 만약 (1) 세 메서드의 정확도-VRAM 파레토 경계가 <10% 범위 내 겹치거나 (2) 최적 메서드가 task 특성과 무관하게 일관성 있게 동일하다면, 검색 메서드 선택은 GPU 제약 환경에서 실무적 의미가 없음.

**최소 실험**: 이틀 실험: (1) 첫날: BM25, dense (FAISS), 하이브리드 재순위 검색 3개 메서드 구현. P1 benchmark에서 3개 representative agent task 선정. 각 메서드 × top-k ∈ {1,3,5,10} = 12개 configuration에 대해 agent 실행 (seed 고정, n_runs=3). 각 실행의 정확도, peak VRAM, 지연시간 기록. (2) 둘째날: 3개 메서드별 파레토 경계 도출 (정확도-VRAM, 정확도-지연시간). 최적 메서드가 task 특성(지식 베이스 크기, 쿼리 복잡도)과 상관성 검증. 결론: 어느 메서드가 어느 조건에서 효율적인가.

**예측**
- 밀집 벡터 검색이 정확도-VRAM trade-off에서 BM25를 평균 >20% 개선하지만, 지연시간에서는 BM25가 우수
- top-k=5 파라미터가 정확도 >85% 달성 시 효율성(VRAM/정확도)에서 top-k=10보다 15% 이상 효율적
- 지식 베이스 크기 >50K 사실에서는 하이브리드 검색(BM25+재순위)이 두 극단 메서드보다 Pareto-optimal 영역 점유

**Novelty 리뷰 (N3: Potentially novel)**: H4 claims to perform empirical quantification of Pareto frontiers (accuracy vs. VRAM/latency) for three retrieval methods under 48GB GPU constraints, mapping task characteristics to optimal method selection. P7's fixture indicates it established retrieval quality as a dominant bottleneck through factorial analysis, but does not explicitly claim: (1) efficiency metric operationalization (VRAM, latency) under hardware constraints, (2) Pareto frontier construction, or (3) task-conditioned method optimality. H4's specific contribution—systematic method comparison (BM25, dense FAISS, hybrid reranking) with dual efficiency objectives under quantified GPU budget—is not attested in P7's description. The hypothesis is internally consistent and feasible (HIGH feasibility rating), but faces critical implementation ambiguities identified in review (baseline fairness across asymmetric methods, embedding model specification, re-ranker choice), which suggest it requires methodological hardening before execution. Given absence of evidence in fixture descriptions that P7 performed GPU-constrained Pareto analysis with task-dependent method selection, H4 qualifies as potentially novel, but the outcome (that retrieval methods exhibit efficiency trade-offs) is somewhat predictable from P7's finding that retrieval quality dominates."

- 가장 가까운 선행연구 [Fixture: Retrieval dominates agent memory bottlenecks](https://arxiv.org/abs/9926.01003), 2026 — 겹침: Factorial study of retrieval methods' effects on agent memory performance; establishes retrieval quality as dominant bottleneck using accuracy metrics / 차이: H4 constructs explicit Pareto frontiers (accuracy-VRAM, accuracy-latency) for method triplets (BM25, dense, hybrid) under GPU constraints and maps task characteristics (knowledge base size, query complexity) to optimal method selection. P7's fixture describes only that 'retrieval quality dominates accuracy,' without clearly operationalizing efficiency metrics (VRAM, latency) under GPU constraints or task-conditional method optimality.
- 가장 가까운 선행연구 [Fixture: Compression helps weak readers more than strong readers](https://arxiv.org/abs/9926.01002), 2026 — 겹침: Studies interaction between method/capability choice and downstream agent performance through reader effects / 차이: H4 focuses on retrieval method efficiency trade-offs under GPU budget; P9 focuses on how reader model strength affects post-retrieval compression gains. Orthogonal contributions.
- 가장 가까운 선행연구 [Fixture: Oracle substitution diagnosis of write and retrieval loss](https://arxiv.org/abs/9926.01004), 2026 — 겹침: Diagnoses retrieval-side performance bottlenecks in agent memory systems / 차이: H4 explicitly rejects oracle substitution diagnosis (P8) as computationally expensive (36+ hours oracle runs) and proposes direct measurement instead (42 hours total for 108 agent runs). This is methodological distinction: measurement approach under resource constraint vs. diagnostic approach without cost operationalization.
- 개선 제안: GPU-constrained retrieval method selection benchmark (BM25, dense FAISS with specification of embedding model, deterministic cross-encoder reranker) constructing Pareto frontiers (accuracy-VRAM, accuracy-latency) on P1 benchmark suite. For each task, identify method + top-k parameter dominating Pareto set. Operationalize baseline fairness explicitly: maximum VRAM envelope per method, latency measured on fixed hardware. Ablate whether agent task success (not just retrieval metrics) is monotonic in retrieval quality via agent trace analysis for random sample of failures. Report embedding model and reranker sensitivity as separate ablation. Expected outcome: task-specific method recommendation table for practitioners deploying under 48GB constraint, rather than general-purpose Pareto frontier claim."

**주요 비판**
- [CRITICAL] Baseline fairness undefined: 48GB GPU budget allocation across BM25 (minimal VRAM), dense FAISS (high VRAM), and hybrid (both) is not operationalized. BM25 index costs ~0 VRAM; FAISS embedding can saturate GPU memory. How is 'equal budget' enforced without disadvantaging inherently asymmetric methods? Running dense on CPU (mentioned as risk mitigation) breaks latency fairness. Requires precise per-method VRAM/latency envelope specification before experiment.
- [CRITICAL] Monotonicity assumption unvalidated: Hypothesis assumes retrieval quality (MRR, top-1 accuracy) monotonically improves agent task success. No mechanism to verify whether the agent actually *uses* or *can exploit* higher-ranked retrieved facts. If agent reasoning is weak or top-1 suffices, further ranking improvements may not transfer to task success. Must include explicit traces showing agent utilization of retrieved facts across methods.
- [MAJOR] Dense retrieval embedding model unspecified: Hypothesis does not fix which sentence-transformer variant, model size, or training data. Choice of embedding model (e.g., small vs. large, domain-specific vs. general) can dominate method performance and VRAM. A smaller embedding model might outperform BM25 on efficiency, but this is embedding choice confound, not retrieval method superiority. Must either fix embedding model or ablate embedding size effects separately.
- [MAJOR] Re-ranker method for hybrid undefined: Hybrid re-ranking's performance is re-ranker-dependent (P3 suggests deterministic freshness; cross-encoder and LLM-based re-rankers are alternatives). Hypothesis does not specify which approach, making 'hybrid' ill-defined. Results may reflect re-ranker choice, not retrieval method trade-offs. Recommend pre-specifying re-ranker (e.g., fixed cross-encoder) or comparing multiple re-rankers as separate methods.
- [MAJOR] Task selection process invites bias: 'Representative 3-4 tasks' is post-hoc criterion. If tasks are chosen after observing preliminary results, overfitting is likely. Recommend: (a) pre-specify task selection rule (e.g., random sample from P1, or all P1 tasks), or (b) pre-register task set before seeing results. With only 3 tasks, generalization claim is weak.


### H2. Runtime-Adaptive Memory Writer Model Scaling Under Extreme Constraints

*Elo 1216 · 2승 1패 0무 · Novelty N2 · Feasibility HIGH · 전략 adaptive*

**가설**: 메모리 쓰기 모델의 크기를 runtime에 동적으로 조정하는 적응형 메타-컨트롤러(1.5B~7B 범위에서 선택)는, 메모리 churn rate와 VRAM 여유도, inference latency를 실시간 모니터링하여 최적 크기를 선택할 때, 고정 크기 baseline 대비 효율성 정규화 성능에서 >8% 개선을 달성할 것으로 가설합니다.

**근거**: P11은 4B 추출 모델이 경쟁적 성능을 보인다고 증명했으나, 극도의 VRAM 제약(48GB) 하에서 작은 모델(1.5B)이 대부분 사용 사례에서 충분할 수 있고, 메모리 volatility가 높은 순간에만 큰 모델(7B)이 필요할 수 있습니다. 현재는 모든 메모리 쓰기에 동일 크기 모델을 사용하므로, 비효율적인 과잉 계산 또는 부정확한 과소 계산이 발생합니다. Runtime metrics를 기반으로 model size를 적응형으로 선택하면, system-wide efficiency를 극대화할 수 있습니다.

**반증 조건**: 만약 adaptive system이 best fixed-size baseline(최적화된 4B 또는 3.5B) 대비 >2% 성능 개선을 달성하지 못하거나, model switching의 오버헤드로 인해 추가 latency spike가 발생한다면 가설은 거짓입니다.

**최소 실험**: 2-3일 실험: (1) 첫날: 1.5B, 2B, 4B, 7B 추출 모델을 고정 크기로 각각 실행하여 F1, VRAM, latency 수집. (2) 둘째날: churn rate 변동 시나리오(10%, 20%, 50% daily update)를 synthetic하게 생성하여 고정 모델 성능 측정. (3) 셋째날: adaptive meta-controller 구현(decision tree) 및 동일 시나리오에서 성능 비교.

**예측**
- Adaptive controller는 48GB VRAM 제약 하에서 최대 95% VRAM 활용도 유지 (고정 4B 모델은 100% 도달 → OOM 위험)
- 메모리 volatility가 낮은 기간(churn rate <10% daily)에서는 1.5B 모델로 switch하여 latency 30% 단축
- 전체 시스템 task success rate는 fixed 4B 대비 >8% 개선 (낮은 volatility에서 더 빠른 업데이트 + 높은 volatility에서도 정확성 유지)

**Novelty 리뷰 (N2: Combination of existing ideas, novelty uncertain)**: H2의 주장은 P11의 기본 발견(소형 모델의 경쟁력)에서 출발하지만, 핵심 기여는 정적 평가에서 동적 적응으로 한 단계 진화시킨 것입니다: (1) 고정 4B 모델 선택 → runtime metrics 기반 적응형 1.5B–7B 선택, (2) 새로운 인과 메커니즘: memory churn rate와 VRAM headroom이 최적 모델 크기를 결정한다는 가설. 

그러나 novelty는 중간 수준(N2)으로 평가됩니다. 이유는: (1) 소형 모델 효율성의 기본은 P11에서 기인, (2) 적응형 선택은 일반적 기법(P12의 test-time scaling과 개념적 유사성)이나 메모리 쓰기에 특화된 탐색 부재, (3) 핵심 가정(churn rate ↔ 최적 모델 크기의 단조 관계)이 검증되지 않은 상태로 제시됨, (4) 리뷰어의 지적: 계산 공정성 문제(1.5B+7B 혼합이 고정 4B 평균보다 높을 수 있음), 의사결정 규칙의 자의성(threshold 설정이 정당화되지 않음). 이러한 요인들이 조합되면, H2는 기존 기법들의 조합이지만 검증이 불충분하고 기초 가정이 미확인된 상태입니다.

- 가장 가까운 선행연구 [Fixture: Small language models as memory writers](https://arxiv.org/abs/9926.01008), 2026 — 겹침: 메모리 쓰기 모델의 크기 효율성 탐색: P11은 4B 모델이 경쟁력 있다는 것을 입증했으며, H2도 동일한 문제 공간(크기 제약 하 메모리 쓰기 효율성)에서 출발합니다. / 차이: P11은 고정된 4B 모델을 정적으로 평가하는 반면, H2는 1.5B–7B 범위에서 runtime에 동적으로 모델을 선택하는 메타-컨트롤러를 제안합니다. H2의 차별점은 시스템 메트릭(churn rate, VRAM headroom, latency)을 기반으로 한 적응형 선택 정책인데, P11에서는 이러한 메커니즘을 제시하지 않았습니다.
- 가장 가까운 선행연구 [Fixture: Test-time compute scaling for scientific hypothesis generation](https://arxiv.org/abs/9925.01011), 2025 — 겹침: Runtime 신호에 기반한 시스템 적응: P12는 test-time compute scaling을 통해 runtime에 계산 자원을 동적으로 할당하는 선례를 보여줍니다. / 차이: P12는 다중 에이전트 generate-debate-evolve 루프의 맥락이며, H2는 사전 학습된 모델 간의 선택 문제입니다. P12는 계산량 증가(test-time scaling)이고, H2는 이미 학습된 모델들 간의 runtime 선택입니다.
- 개선 제안: 더 강력한 버전: '동일한 계산 예산 제약 하에서, memory churn rate와 VRAM headroom을 기반으로 한 runtime-adaptive model selection은 고정 크기 모델(동일 평균 compute 예산)보다 >8% task success rate 개선을 달성한다. 핵심 메커니즘: 저 churn 상황에서는 1.5B 사용(latency 감소), 고 churn 상황에서는 4B/7B 사용(정확성 유지).' 이 버전은 (1) compute-matched baseline을 명시, (2) churn rate의 구체적 임계값을 실험 결과로 유도(사전 가정 아님), (3) adaptation의 정확한 의사결정 규칙을 ablation으로 검증, (4) 각 churn 레벨별 성능 특성화를 분리된 실험으로 제시하면 더 견고합니다.

**주요 비판**
- [CRITICAL] Baseline fairness violation: The adaptive system switches between 1.5B and 7B models. If it allocates 50% compute to 7B (average >4B per call), this is not a fair comparison to fixed 4B. Any performance gain might reflect higher compute budget, not better adaptation. Comparison should use compute-matched baselines (e.g., rotation between 1.5B and 4B), which are listed but not clearly established as primary baseline.
- [CRITICAL] Unvalidated core assumption: Hypothesis assumes 'monotonic relationship' between memory churn rate and optimal writer model size. No empirical evidence provided. Churn rate (update frequency) may not correlate with optimal model size if the constraint is accuracy, not speed. High-churn scenarios might need larger models for accuracy, but this is presented as an assumption, not a finding.
- [MAJOR] Decision logic lacks justification: Proposed heuristic thresholds (churn_rate >20%, vram_free >5GB → 7B) appear arbitrary. No ablation or tuning strategy specified. If thresholds are tuned on the test task, this introduces leakage. If tuned on separate validation data, 2-3 day timeline does not permit this.
- [MAJOR] Model switching overhead underestimated: Hypothesis claims overhead is 'negligible' but loading different model weights into VRAM, cache invalidation, and warmup have real latency cost. Falsification criterion allows ±2% performance or 'latency spike,' but doesn't specify threshold. Frequent switching could introduce >100ms overhead, making system worse, not better.
- [MAJOR] Novelty claim overstated: P11 already establishes that smaller (4B) models are competitive. Adaptive selection itself is standard engineering (mixture of experts, dynamic model switching). Claimed novelty ('1.5B-7B characterization' + 'runtime adaptation') requires evidence that: (1) 1.5B actually works for memory extraction [P11 does not validate this], and (2) runtime adaptation of writer size is unexplored. Neither is demonstrated.


### H1-v2. 경량 진단으로 제약 환경에서 검색 병목 검증

*Elo 1199 · 1승 1패 0무 · Novelty N2 · Feasibility HIGH · 전략 feasibility*

**가설**: Write-retrieval 손실을 상호작용-인식 약한 오라클 진단(15-25개 고분산 케이스) 및 P5/P6 기반 분석적 상한으로 진단하면, <3일 실험에서 검색이 에이전트 메모리 정확도의 주요 병목(기여도 >60%, 95% 신뢰구간)임을 확인할 수 있으며, 이는 제약 환경에서 P7의 retrieval dominance를 실증적으로 검증함과 동시에 후속 검색 우선 최적화 가설을 정당화합니다.

**근거**: P7은 retrieval 품질이 정확도를 지배함을 보였으나, 무제한 API와 충분한 VRAM 환경에서의 결과입니다. 극도의 GPU 제약에서는 (1) write-retrieval 상호작용이 더 강해지만 가능성(P5 간섭, P6 시간 정보 손실의 복합 효과), (2) 메모리 용량 부족으로 인한 선택적 압축이 write와 retrieval에 다른 영향. H1의 비용 문제(150+ GPU 시간)를 해결하면서도 bottleneck을 신뢰도 높게 식별하려면, (a) 높은 정보 가치의 작은 샘플 선택(고분산 시나리오), (b) 분석적 상한으로 보간 비용 절감, (c) 상호작용 효과 명시적 모델링이 필수입니다. 이 과정에서 제약 환경이 retrieval 병목을 강화하는지 약화하는지 실증적으로 검증할 수 있습니다.

**반증 조건**: 만약 write-loss upper bound >40% (간섭 조정 후도), 또는 oracle숱 analytical bounds 편차 >25%, 또는 retrieval loss가 명확한 병목이 되지 않음(신뢰 구간 <55%), 가설은 거짓됩니다.

**최소 실험**: 2-3일 실험: (1) 1일차 오전: P1 benchmark에서 5개 대표 태스크 선택 기준 정의(검색 품질 표준편차, 충돌 빈도), 각 태스크에서 3개 오라클 케이스(perfect write+imperfect retrieval, imperfect write+perfect retrieval, both perfect) 구성. 총 15개 오라클 실행. (2) 1일차 오후: P5(간섭 팩터) + P6(시간 정보 손실)로부터 oracle 결과에 대한 analytical bounds 도출. oracle vs bounds 편차 측정. (3) 2일차: write-loss upper bound, retrieval-loss lower bound 계산. bottleneck 기여도 추정 및 95% 신뢰구간 도출(bootstrap). (4) 3일차(필요시): 3개 정규화 방식으로 결론 robust 확인. 최종 report: 검색 병목 기여도와 신뢰도.

**Novelty 리뷰 (N2: Combination of existing ideas, novelty uncertain)**: H1-v2는 P7의 검색 주도 병목 발견이 GPU 제약 환경에서 일반화되는지 테스트하는 비용 효율적인 진단 접근을 제안합니다. 핵심 신규성 주장은 경험적입니다: 검색이 쓰기 측 메모리 압박과 추론 지연 제약 하에서도 주요 병목으로 남아 있다는 것. 그러나 실질적 차이는 세 가지 요인으로 제한됩니다: (1) **방법론적 비신규성**: 오라클 접근은 P8에서, 분석적 경계는 P5/P6에서 기원하며, 이들을 결합하는 것은 새로운 응용일 뿐 방법론적 혁신이 아닙니다. (2) **불확정적 경험적 결과**: P7은 무제약 설정에서 수행됨. 검색 주도성이 GPU 제약 하에서 **지속되는지**는 개방된 질문이지 확인된 발견이 아닙니다. 가설은 거짓될 수 있습니다 (쓰기 측 결정이 메모리 압박 하에서 중요해질 수 있음). (3) **표본 크기 및 일반화 위험**: n=15–25 사례(5개 태스크)는 작음. 병목 기여도 신뢰도는 고분산 시나리오 선택이 편향을 도입하는지, 분석적 경계가 제약 조건 하에서 유효한지(리뷰 비판에서 명시적으로 표시됨)에 따라 결정됩니다. 신규성은 경험적 결과가 확인적(검색 유지)이고 진단 방법이 제약 조건 하에서 타당한 경우에만 조건부입니다. N2 분류는 이 불확실성을 반영합니다: 기존 방법(P8 + P5/P6)을 새로운 체제(GPU 제약) 하에서 확립된 가설(P7)을 테스트하는 데 적용하는 조합은 신규적으로 확립되지도 파생적으로 명확하지도 않습니다—경험적 결과와 방법론적 견고성에 따라 결정됩니다.</rationale>
<parameter name="remaining_uncertainty">1. P7의 검색 주도성 순위가 실제로 GPU 제약 하에서 지속되는지, 아니면 쓰기 측 결정(압축, 제거, 선택적 저장)이 지배적 또는 공동 지배적이 되는지. 2. P5(간섭 ×0.7–0.8)와 P6(시간 손실 ~10–15% 검색 성능 저하)에서 도출된 분석적 경계가 GPU 제약 하에서 독립적으로 유효하고 올바르게 구성되는지. 오라클 결과에 대한 사전 검증이 중요함. 편차 >25%면 비용 절감 전략이 무효화됨. 3. 오라클 표본 크기(n=15–25)와 정의되지 않은 '고분산' 기준에 따른 층화 선택이 충분한 선택 편향을 도입하여 거짓 병목 기여도를 야기하는지. 95% CI에서의 부트스트랩은 작은 유효 표본 크기를 고려할 때 신뢰도를 과표현할 수 있음. 4. 오라클 분해가 쓰기 vs. 검색 손실을 깔끔하게 격리하는지, 또는 오라클 대체 하에서 시스템 수준 성능을 혼동하는지(리뷰에서 인정한 혼동). 5. 제약 체제 진단이 진정으로 검색 우선 최적화 가설을 정당화하는지, 아니면 제약이 다른 인과 구조를 드러내는지(예: 쓰기-검색 공동 최적화가 필수가 됨).</remaining_uncertainty>
<parameter name="suggested_reformulation">정규화 제안: (1) **경험적 초점 명확화**: 'H1-v2는 P7의 검색 주도 병목 발견이 GPU 제약 환경에 일반화되는지 검증하며, 만약 제약 조건에서 기여도 순위가 역전되면 이는 고도로 새로운 발견입니다'라고 명시. (2) **경합 가설 명시**: (a) 검색 지배적 (P7 확인), (b) 쓰기 지배적 (제약이 우선순위 역전), (c) 상호작용 필수 (쓰기+검색 공동 최적화)의 세 시나리오를 구분. (3) **진단 방법 한계 노출**: 오라클 표본 크기 재검토(n≥50 권장) 및 분석적 경계 사전 검증 게이트 추가('oracle vs. bounds 편차 >25%면 외삽 중단'). (4) **비용-신뢰도 tradeoff 정량화**: '3일 실험으로 60% 기여도를 95% CI로 검증 vs. 기여도 ±15% 불확도 수용 시 계산 절감' 명시적 선택. (5) **원점 명확화**: H1-v2의 성공은 '검색이 여전히 주요 병목'이라는 P7 지지이지만, 실패는 '제약이 병목을 재구성'이라는 새로운 발견. 프레임을 이 이중 결과로 재구성하면 N2→N3 전환 가능성 증가.</suggested_reformulation>
<parameter name="substantive_difference">true

- 가장 가까운 선행연구 [Fixture: Retrieval dominates agent memory bottlenecks](https://arxiv.org/abs/9926.01003), 2026 — 겹침: Both test whether retrieval is the dominant bottleneck limiting agent memory accuracy. Both employ empirical factorial-style methodologies to isolate write-side vs. retrieval-side losses. Both target the same research question: which subsystem (write or retrieval) most constrains performance. / 차이: P7 operates under unrestricted API access and sufficient VRAM. H1-v2 tests the same hypothesis under strict GPU memory constraints (RTX A5000 ×2, ~48GB). H1-v2 uses oracle sampling + analytical bounds (from P5/P6) to reduce cost from ~150 GPU-hours to ~40–50. P7 does not investigate whether bottleneck ranking persists under write-side memory pressure or inference latency constraints.
- 가장 가까운 선행연구 [Fixture: Oracle substitution diagnosis of write and retrieval loss](https://arxiv.org/abs/9926.01004), 2026 — 겹침: Both employ oracle substitution to diagnose write-side vs. retrieval-side losses. Oracle configurations (perfect write + imperfect retrieval, vice versa, both perfect) are directly adopted from P8's methodology. Both use oracle evidence to infer system bottlenecks. / 차이: P8 provides the diagnostic framework but does not apply it to GPU-constrained settings or combine it with analytical bounds. H1-v2 explicitly couples P8's oracle diagnostics with cost-reduction via analytical interpolation (P5/P6 multipliers) and targets a specific empirical question under resource scarcity. P8 does not test the persistence of P7's retrieval-dominance finding.
- 가장 가까운 선행연구 [Fixture: Interference between conflicting memories in continual agents](https://arxiv.org/abs/9926.01009), 2026 — 겹침: P5 quantifies interference effects (plasticity multiplier ~0.7–0.8×) on retrieval-based memory. H1-v2 uses this factor as an analytical bound to adjust oracle diagnostics, treating interference as a known loss mechanism. / 차이: P5 studies interference in isolation; H1-v2 treats it as one component of a composite analytical upper bound for bottleneck diagnosis. P5 does not address how interference interacts with GPU memory constraints or whether interference ranking changes under constrained regimes.

**주요 비판**
- [CRITICAL] [P7] demonstrated retrieval dominance under *unrestricted* API and VRAM. H1-v2 assumes this ranking persists in GPU-constrained regimes, but this is unargued. Under memory pressure, write-side decisions (what to keep, compress, evict) may become critical or dominant, potentially reversing the bottleneck. The hypothesis should either (a) theoretically justify why retrieval must remain dominant despite constraints, or (b) frame this as a competing-hypotheses test with explicit predictions for each scenario (retrieval-dominant vs. write-dominant vs. parity).
- [CRITICAL] Oracle sample size (n=15–25, from 5 tasks × 3 cases) is extremely small. Stratified selection on 'high-variance' scenarios (metadata-based filtering never formally defined) introduces selection bias. Even with bootstrap CI and i.i.d. assumption, a single anomalous task could shift bottleneck attribution ±15–20 percentage points. Either increase sample to n≥50–75 (cost-feasible) or explicitly accept posterior uncertainty and lower confidence threshold from 95% CI to 80%.
- [CRITICAL] Analytical bounds (P5 plasticity 0.7–0.8×, P6 temporal loss 10–15% retrieval degradation) are treated as fixed universal multipliers, but P5 and P6 are empirical findings from other contexts. No pre-validation that these multipliers hold under GPU constraints, compose correctly (independence assumption), or generalize to the task distribution. If oracle vs. analytical bounds diverge >25%, the bounds-based diagnosis fails. Recommend validating bounds against oracle results as a gate experiment before using them for extrapolation cost-reduction.
- [MAJOR] Oracle decomposition does not cleanly isolate write vs. retrieval loss. The oracle configurations (perfect write + imperfect retrieval vs. imperfect write + perfect retrieval) swap subsystems rather than isolating independent contributions. In practice, write quality shifts the distribution of available facts, affecting retrieval's operating point. Reference to [P8] (oracle substitution) acknowledges this but does not resolve it. The inferred bottleneck (e.g., 'retrieval loss = performance gap under oracle retrieval') conflates two different systems, not isolated components. Consider framing as 'system performance under oracle retrieval subsystem' vs. 'under oracle write subsystem' with explicit discussion of this confound.
- [MAJOR] Bottleneck definition may conflate multiple constraints. [P7] identified retrieval as bottleneck for *accuracy* under unlimited compute. GPU constraints introduce latency and memory-footprint constraints; write operations (compression, eviction) may dominate latency/footprint while retrieval dominates accuracy. The hypothesis must explicitly define bottleneck scope (accuracy vs. latency vs. memory) and validate that metrics align with this definition.


### H1. Efficient Oracle Substitution for Memory Bottleneck Decomposition

*Elo 1199 · 1승 1패 0무 · Novelty N1 · Feasibility HIGH · 전략 measurement*

**가설**: 우리는 oracle substitution 진단을 극도의 GPU 제약(48GB) 환경에 최적화하면, 메모리 시스템의 write-side loss와 retrieval-side loss를 신뢰할 수 있게 분해할 수 있으며, 계산 예산으로 정규화한 효율성 지표가 raw 성능 랭킹과 >20% 이상 다른 Pareto frontier를 드러낼 것으로 가설합니다.

**근거**: P8은 oracle substitution을 이론적으로 제시했으나 무제한 API 접근과 상대적으로 충분한 VRAM을 가정했습니다. 극도의 제약에서는 oracle 구성 자체가 높은 계산 비용이므로, 효율적인 샘플링 및 보간 기법이 필요합니다. 또한 P2에서 LLM-as-judge가 biased된 점수를 생성하므로, oracle substitution 기반 평가는 더 신뢰할 수 있는 벤치마킹을 제공할 수 있습니다. 효율성 정규화(efficiency-normalized performance)는 메모리-VRAM-latency trade-off 공간에서 실제 실용적 우위 전략을 식별하게 됩니다.

**반증 조건**: 만약 oracle substitution과 LLM-as-judge의 랭킹이 Spearman correlation >0.9 이상이고, 효율성 정규화가 랭킹을 <10% 변경한다면 가설은 거짓입니다.

**최소 실험**: 이틀 실험: (1) 첫날: 3개 메모리 전략에 대해 oracle case 50개 구성 및 linear interpolation으로 나머지 150개 추정. LLM-as-judge 평가 병행. (2) 둘째날: write-loss와 retrieval-loss 상한 계산 및 효율성 정규화 점수 도출. Oracle과 LLM-as-judge 간 correlation 측정 및 신뢰도 분석.

**예측**
- Retrieval-dominated bottleneck이 oracle substitution으로 test case의 >75%에서 식별되며, 이는 P7 결론과 일관성 있음
- 효율성 정규화 후 전략 랭킹이 raw 성능 랭킹과 >25% 재정렬
- LLM-as-judge 점수가 oracle substitution 점수와 >45% case에서 Spearman correlation <0.8 수준 차이

**Novelty 리뷰 (N1: Minor variation of known work)**: H1은 P8의 oracle substitution 방법론을 GPU 제약 환경에 최적화하고 효율성 정규화 메트릭을 추가한 것으로, 근본적으로 새로운 연구 기여라기보다 기존 방법론의 공학적 개선입니다. 주장된 혁신: (1) 자원 제약 최적화(P8을 48GB VRAM에 적용)—표준 시스템 공학 작업이지 새로운 방법이 아님. (2) 효율성 정규화 메트릭—시스템 연구의 표준 관행(성능/전력 비율, 처리량/지연시간 트레이드오프). (3) oracle 비용 감소를 위한 샘플링-보간—새로운 계산 기법이나, 비선형 agent 메모리 손실 환경에서 체계적 편향을 도입하여 과학적 타당성이 의심됨. 핵심 지적 내용(oracle substitution 진단, write vs. retrieval 분해)은 P8에서 변하지 않음. H1은 새로운 이론적 통찰, 새로운 메모리 메커니즘, 또는 메모리 시스템이 왜 작동하는지에 대한 실증적 발견을 주장하지 않고—오직 기존 진단(P8)을 제약 환경에 적용하고 다른 평가 메트릭을 사용할 수 있음을 보일 뿐입니다. 리뷰의 "주로 방법론적 리팩토링이지 실증적 발견이 아님" 분류는 N1(경미한 변형)과 일치합니다. 추가로, 가설의 독립성 가정(write와 retrieval 손실이 독립적이거나 약하게 상호작용)은 P5(메모리 간섭으로 인한 가소성 감소)와 P6(write 시점의 시간적 단서 손실)에 직접 모순되어 oracle 분해의 타당성을 훼손합니다. 이는 가설에서 다루어지지 않아 신규성 주장을 더욱 약화시킵니다.

- 가장 가까운 선행연구 [Fixture: Oracle substitution diagnosis of write and retrieval loss](https://arxiv.org/abs/9926.01004), 2026 — 겹침: Core diagnostic method: oracle substitution with evidence substitution for write-loss and retrieval-loss decomposition. Both use oracle construction (perfect write, perfect retrieval) as ground truth to bound individual loss components. / 차이: H1 optimizes P8's method for extreme GPU constraints (48GB VRAM) and adds efficiency-normalized metrics (cost-adjusted Pareto frontier). P8 assumes unlimited API access and sufficient computational budget. H1 proposes sampling-interpolation to reduce oracle construction cost.
- 가장 가까운 선행연구 [Fixture: LLM-as-judge leniency on memory benchmarks](https://arxiv.org/abs/9926.01010), 2026 — 겹침: Both identify unreliability in LLM-as-judge evaluation. H1 uses oracle substitution as alternative to address P2's leniency problem. / 차이: P2 diagnoses the bias (judges accept vague answers). H1 proposes oracle substitution as a more trustworthy evaluation protocol, but does not introduce a fundamentally new diagnostic method—it applies P8's existing technique.
- 가장 가까운 선행연구 [Fixture: Retrieval dominates agent memory bottlenecks](https://arxiv.org/abs/9926.01003), 2026 — 겹침: Both treat write vs. retrieval loss decomposition as scientifically important. P7 provides empirical evidence that retrieval dominates; H1 uses this to motivate oracle substitution for bottleneck diagnosis. / 차이: P7 is an empirical finding via factorial study. H1 is a methodological proposal for more precise diagnosis. H1 does not claim new empirical mechanisms beyond what P7 reports.
- 개선 제안: H1을 본질적 신규성 주장으로 재구성: (1) **write-retrieval 손실 결합 모델 제안**—P5/P6 상호작용을 명시적으로 다룸(예: write 시점 정규화가 retrieval 시점 시간 추론을 감소시킴, 또는 공유 메모리 용량이 write-retrieval 트레이드오프 생성). oracle substitution이 이 결합을 고려해야 함을 보임. (2) **새로운 평가 프로토콜 주장**: '…효율성 정규화 oracle substitution + LLM-as-judge 교차 검증은 메모리 전략의 견고하고 메커니즘적으로 해석 가능한 랭킹을 제공함.' 효율성 가중치를 사전에 커밋(예: F = task_accuracy / (peak_VRAM · latency_penalty))하고 이론적으로 정당화. (3) **실증적 발견**: '효율성 정규화 후, retrieval 최적화 전략은 ≤48GB 제약 하에서 hybrid write-retrieval 트레이드오프 전략으로 지배당하며, P7의 retrieval 지배 발견이 계산이 무제한일 때만 일반화됨을 드러냄.' 결과가 성립하면 N3. (4) **보간 검증**: linear interpolation이 held-out 검증 집합에서 실제 oracle 값과 >0.9 상관관계를 보임을 주요 실험 전에 보임. 보간이 검증에 실패하면 편향을 명시적으로 인정하거나 폐기.

**주요 비판**
- [CRITICAL] Independence assumption violated. Hypothesis assumes write and retrieval loss are 'independent or weakly interacting,' but P5 (memory interference reducing update plasticity) and P6 (write-time fact extraction dropping temporal cues that hurt later retrieval) directly contradict this. Decomposition via oracle substitution assumes these losses can be isolated; interaction effects mean oracle bounds may not accurately diagnose bottlenecks. Requires explicit modeling of write-retrieval coupling before experiments.
- [MAJOR] Efficiency normalization scheme under-specified. Hypothesis claims >25% reranking after normalization but does not commit to how to weight peak VRAM, latency, token budget, and task accuracy into a single score. Different weighting schemes (linear, logarithmic, weighted sum) produce different Pareto frontiers. Without pre-specifying this before running experiments, results appear post-hoc and arbitrary.
- [MAJOR] Compute budget risk. Minimum experiment (48 hours) is optimistic. Constructing 50 oracle cases (perfect write + perfect retrieval) for each strategy requires separate agent runs. With 3+ strategies, multiple random seeds, and validation, total cost could reach 150+ GPU hours. RTX A5000 throughput (~24 TFLOPS) means ~7 days wall-clock minimum. Interpolation of missing 150 oracle values via linear interpolation is analytically weak: agent memory loss landscapes are non-convex; interpolation introduces systematic bias.
- [MAJOR] Limited scientific novelty. Hypothesis re-applies P8's oracle substitution (existing method) with standard efficiency metrics (from systems engineering) and Spearman correlation (standard statistics). Components are not novel; combination lacks theoretical insight. Positioned as enabling 'practically superior strategies' but is primarily a methodological refactoring, not an empirical discovery about memory mechanisms.
- [MAJOR] Scope ambiguity. Research goal asks: compare long-term memory strategies under constraints. H1 proposes: develop a better evaluation protocol. These are different research goals. If oracle substitution ranking = LLM-as-judge ranking (likely given P7's retrieval dominance is consistent across methods), then protocol change reveals nothing new about strategies themselves.


### H3. Integrated Memory Lifecycle: Coordinated Replay, Conflict Resolution, Compression

*Elo 1153 · 0승 3패 0무 · Novelty N2 · Feasibility MEDIUM · 전략 cross_subfield_combination*

**가설**: 우리는 경험 재생(stage 1: raw experience capture, P4에서), 결정론적 conflict resolution(stage 2: consolidation, P3에서), 점진적 압축(stage 3: archival, P9에서)을 통합한 메모리 수명 관리 프레임워크가, 각 메커니즘을 독립적으로 적용하는 것 대비 메모리 효율성(quality-per-VRAM-per-token)을 25-35% 개선하고, 도메인 간 일반화를 >80% 달성할 것으로 가설합니다.

**근거**: 기존 연구는 각 메커니즘을 독립적으로 평가합니다: P4는 경험 재생을 웹 네비게이션에만 적용, P9는 압축을 단독으로 분석, P3, P5는 conflict resolution을 고립된 설정에서 평가. 그러나 실제 continual agent 환경에서 메모리는 lifecycle을 거칩니다: (1) 새로운 상호작용을 raw experience로 캡처 → (2) 기존 메모리와의 충돌 해결 (update plasticity) → (3) 시간 경과 또는 VRAM 압박 하에서 점진적 압축. 이 세 단계를 분리하면 정보 손실(P6 temporal loss during write, P9 compression-induced loss)이 누적되고, 각 단계의 최적화가 다음 단계의 성능을 악화시킬 수 있습니다. 통합 lifecycle 설계는 이러한 단계 간의 정보 손실을 최소화하고, 일관성 있는 compression schedule 및 conflict resolution 정책을 적용하여 domain-agnostic하게 작동할 수 있습니다.

**반증 조건**: 만약 (1) lifecycle 통합이 메모리 효율성을 15% 이상 개선하지 못하거나, (2) domain generalization이 in-distribution 대비 <60% 성능 달성하거나, (3) compression과 conflict resolution의 조합이 각 메커니즘의 독립 성능 합 이상을 달성하지 못한다면 가설은 거짓입니다.

**최소 실험**: 3-4일 실험: (1) 첫날: experience replay capture 구현 및 3개 도메인에서 raw memory collection. (2) 둘째날: consolidation stage(conflict resolution) 구현 및 conflict type 분류 + 메모리 accuracy 측정. (3) 셋째날: progressive compression schedule 구현 및 temporal loss 측정. (4) 넷째날: 3가지 도메인에서 통합 lifecycle 성능 평가 및 baseline 비교.

**예측**
- Lifecycle 통합 시스템은 메모리 footprint를 단계별 독립 실행 대비 25-35% 감소 시키면서 accuracy >90% 유지
- Web navigation domain에서 >95% in-distribution 성능 달성, dialogue/planning domain에서 >80-85% transfer 성능 달성 (vs. 단일 메커니즘 <70%)
- Conflict resolution + compression의 조정으로 temporal information loss(P6)를 30% 이상 감소
- Consolidation stage에서 deterministic conflict resolution(P3)이 LLM-based resolution보다 10% 빠르면서 accuracy 동등 또는 우수

**Novelty 리뷰 (N2: Combination of existing ideas, novelty uncertain)**: H3는 세 가지 기존 메커니즘(P4의 경험 재생 → P3의 충돌 해결 → P9의 압축)을 시간 순서로 배열하고 통합 이점(25–35% 효율성 개선, >80% 도메인 간 전이)을 주장합니다. 그러나 본질적 신규성은 제한적입니다:

1. **메커니즘 출처**: 세 메커니즘 모두 기존 논문(P4, P3, P9)에서 나옴. H3은 새로운 메커니즘을 기여하지 않으며, 시간적 순서와 주장된 조정만 제공.

2. **통합을 공학 vs 신규성으로 봄**: 수명주기 단계(캡처 → 통합 → 보관)는 자연스러운 시스템 설계 패턴이며, 개념적 진전이 아님. 새로운 알고리즘이나 이론적 기여 없이 기존 요소를 결합하는 것은 경미한 변형.

3. **정당화되지 않은 개선 주장**: 25–35% 효율성 목표는 정량적 도출이 부재. P6은 ~10% 시간적 손실을 식별; P5는 고립된 설정에서 충돌 유도 손실 표시; P9는 더 강한 모델에서 압축 이득 감소 보임. 이들이 25–35%에 합산되지 않음. 숫자가 자의적으로 보임.

4. **치명적 검색 병목 모순**: P7은 인수분해 분석을 통해 검색 품질이 에이전트 정확도를 지배한다고 명시. H3는 쓰기측 메커니즘(캡처, 통합, 압축)만 다루며 검색 개선을 제안하지 않음. 이는 선행 연구에서 식별한 결정적 제약과 근본적 불일치.

5. **근거 없는 도메인 간 일반화**: 웹 네비게이션(클릭 시퀀스), 대화(발화), 코드 생성(컨텍스트), 계획(상태 표현) 간 >80% 전이를 주장하는 것은 이들 도메인이 요구하는 다양한 메모리 형식과 모순. 선행 연구(P4, P3, P5, P9) 중 어느 것도 이 네 도메인에서 >80% 전이 성공을 입증하지 않음. 주장이 추측적.

6. **해결되지 않은 기준선 공정성**: 통합 수명주기는 공동 튜닝과 최적화를 받을 것이며, 주장된 '독립적' 기준선(P4, P3, P9 별도 적용)은 선행 연구에서 공동 튜닝되지 않음. '통합이 도움됨'과 '공동 최적화가 도움됨'을 혼동. 통합 주장을 검증하기 위해 필수적인 제어—세 메커니즘 공동 튜닝되지만 모듈식 유지—존재하지 않음.

- 가장 가까운 선행연구 [Fixture: Experience replay memory for LLM agents in web tasks](https://arxiv.org/abs/9924.01007), 2024 — 겹침: Experience replay mechanism for capturing agent trajectories as reusable memory artifacts / 차이: H3 extends P4's web-only application to dialogue, code generation, and planning domains; P4 does not integrate with consolidation or compression stages
- 가장 가까운 선행연구 [Fixture: Deterministic freshness resolution for memory conflicts](https://arxiv.org/abs/9926.01006), 2026 — 겹침: Deterministic conflict resolution using serial numbers to select latest versions during memory updates / 차이: H3 embeds P3's mechanism as stage 2 of a lifecycle, claiming it reduces temporal information loss from write-time fact extraction (P6); P3 tested conflict resolution in isolation without analyzing temporal loss propagation
- 가장 가까운 선행연구 [Fixture: Compression helps weak readers more than strong readers](https://arxiv.org/abs/9926.01002), 2026 — 겹침: Progressive compression of stored memories to reduce VRAM footprint while maintaining accuracy / 차이: H3 coordinates P9's compression schedule with P3's conflict resolution to preserve temporal metadata; P9 analyzes compression independently without addressing interaction with conflict resolution or write-time loss (P6)
- 가장 가까운 선행연구 [Fixture: Retrieval dominates agent memory bottlenecks](https://arxiv.org/abs/9926.01003), 2026 — 겹침: Factorial study of memory mechanisms identifying performance bottlenecks in agent systems / 차이: P7 concludes retrieval quality dominates agent accuracy; H3 focuses exclusively on write-side optimization (capture, consolidation, compression) and does not propose retrieval improvements—critical mismatch given P7's finding
- 개선 제안: 치명적 위협을 다루도록 재구성: (1) 쓰기측 vs 검색측 손실을 측정하는 데 P8의 오라클 대체 방법론 사용—검색 손실이 지배적인 경우(P7에 따름) 그곳에 노력 집중. (2) 독립 메커니즘을 공정하게 공동 최적화하는 기준선 설계(P4 + P3 + P9, 공유 계산 예산, 함께 튜닝되지만 모듈식 유지)하여 통합 자체의 기여 분리. (3) 자의적 25–35% 목표를 구성 요소 수준 분석으로 대체: P6의 시간적 손실 정량화, 제어된 설정에서 충돌 해결 이점 측정, P9 스케일링 법칙에 따른 압축 이득 추정—기대 개선도를 상향식으로 도출. (4) 웹 네비게이션(P4 성공한 도메인)부터 시작하고 정확한 전이 성능(분포 내 vs 분포 외 정확도) 측정 후 >80% 도메인 간 전이 주장. (5) 단일 통합 통찰(예: '압축 중 시간적 메타데이터 보존이 최근 결정의 검색 개선') 명시하고 이를 테스트하는 제거 실험 설계—모호한 '조정'으로부터의 이점 주장 대신."

**주요 비판**
- [CRITICAL] Baseline fairness uncontrolled: integrated lifecycle will receive joint tuning and optimization, while 'independent' baselines (P4, P3, P9 applied separately) were not co-tuned. This conflates 'integration helps' with 'joint tuning helps.' A jointly-tuned independent variant (three mechanisms with shared compute budget, optimized together but kept modular) is essential to verify the integration claim.
- [CRITICAL] Retrieval bottleneck dominance [P7] undermines entire hypothesis: P7 states 'retrieval quality dominates accuracy' in agent memory systems. Hypothesis focuses on write-side (capture, consolidation, compression) but does not propose retrieval improvements. If retrieval loss >> write-side loss, the three mechanisms cannot fix the binding constraint. Oracle experiments [P8] to measure write vs. retrieval loss should precede this hypothesis.
- [MAJOR] Claimed improvements (25–35%) lack justification from prior work: P6 warns of ~10% temporal loss during write, P5 shows conflict-induced loss in limited scenarios, P9 shows compression gains that decrease with stronger models. These do not sum to 25–35% improvement. The number appears arbitrary and overstates expected benefit without quantitative derivation.
- [MAJOR] Domain generalization claim (>80% transfer) is unsupported and vague: Web navigation (P4), dialogue, code generation, and planning have incompatible memory formats (click sequences, utterances, state representations, code contexts). A 'domain-agnostic core format (timestamp, entity, action, outcome)' loses critical domain-specific structure. No prior work demonstrates successful transfer across these four domains; 80% is unanchored.
- [MAJOR] Novelty is weak: combining P4 (replay), P3 (conflict resolution), P9 (compression) without algorithmic innovation is obvious engineering, not research contribution. Hypothesis claims P4/P9/P3 did not test integration, but absence of evidence is not evidence of novelty. Contribution should either propose novel integration mechanism (e.g., temporal-aware compression through conflict metadata) or reframe as pure empirical baseline paper.


## 지식 공백 (literature digest)

- **G1** (untested assumption) — 모르는 것: RTX A5000 2장(~48GB VRAM) 같은 극도로 제한된 GPU 자원 환경에서 메모리 관리 전략별 성능-효율성 trade-off의 정량적 특성화. 동일한 계산 예산(예: 100M tokens/월)으로 정규화했을 때 task success rate, peak VRAM usage, inference latency, memory retrieval accuracy의 Pareto frontier는 무엇인가? · 검증: 동일 에이전트 아키텍처에서 메모리 전략별 구현 (경험 재생 vs 결정론적 conflict resolution vs 압축 vs 작은 추출 모델 vs 이들의 조합), 동일 계산 예산 제약 하에서 peak VRAM, end-to-end latency, task completion rate 측정, 전략별 cost-benefit curve 도출 · 이미 풀렸을 위험: 중간. 각 메모리 관리 기법은 이미 알려졌으나, compute-constrained 환경에서의 체계적 비교 연구는 부재할 가능성 높음. 기존 연구가 대규모 리소스를 가정했으므로 novelty risk 중간 수준. · 근거: P4, P3, P9, P11
- **G5** (untested assumption) — 모르는 것: 메모리 쓰기 모델 크기(1B vs 2B vs 3.5B vs 4B vs 7B)가 1) 메모리 추출 정확도, 2) peak GPU memory usage, 3) inference latency, 4) downstream agent task success rate에 미치는 영향의 정량적 trade-off. 극도의 VRAM 제약(48GB) 하에서 최적 모델 크기는 무엇인가? · 검증: 동일 에이전트 환경에서 다양한 크기의 미세조정 추출 모델(1B, 1.5B, 2B, 3.5B, 4B, 7B) 통합 → 메모리 추출 F1 score, memory-augmented agent task success rate, peak VRAM, inference latency 측정 → 크기별 cost-performance curve 도출 → Pareto optimal 크기 식별 · 이미 풀렸을 위험: 중간. P11은 4B의 가능성을 보였으나, 1B~7B의 체계적 비교와 극도의 리소스 제약 환경에서의 최적화는 미측정. 모델 스케일 분석은 흔한 연구 주제이나, 메모리 시스템 특화 환경에서의 ablation은 novel. · 근거: P11
- **G2** (contradictory findings) — 모르는 것: write-time information loss (P6)와 retrieval-phase bottleneck (P7) 각각이 최종 에이전트 성능에 기여하는 상대적 크기. 리소스 제약 하에서 어느 단계를 먼저 최적화할 것인가? 예를 들어, 쓰기 정확도 10% 개선 vs 검색 recall 10% 개선 중 어느 것이 더 큰 성능 이득을 가져오는가? · 검증: P8의 oracle substitution 기법 활용: 1) perfect write를 가정한 retrieve-only 성능 측정 (write-loss의 상한 추정), 2) perfect retrieval을 가정한 write-only 성능 측정 (retrieval-loss의 상한 추정), 3) 가우스 소거법으로 상대적 영향도 정량화. 또는 ablation으로 write quality를 단계적으로 degradation하면서 성능 변화 측정. · 이미 풀렸을 위험: 중간. P8이 진단 방법론을 제시했으나, 극도의 리소스 제약 환경에서의 상대적 영향도 분석은 미측정. oracle substitution을 극도의 constraint 하에서 효율적으로 수행하는 것이 novelty 포인트. · 근거: P6, P7, P8
- **G4** (evaluation weakness) — 모르는 것: 메모리 압축의 정량적 계산 비용(토큰, VRAM, latency)과 이득(quality improvement)의 cost-benefit trade-off. 만약 1% accuracy gain을 위해 20% 압축 지연시간이 필요하다면, RTX A5000 2장 환경에서 압축은 부정적 수익성을 가진다. 각 모델 크기별로 압축의 순 효율성(NPS = quality_gain - cost_penalty)은 무엇인가? · 검증: 다양한 압축 알고리즘(summarization, token pruning, clustering)과 모델 크기(1.5B~7B readers)별로 1) 압축 계산 시간/토큰/VRAM, 2) 압축 후 memory retrieval F1 score 측정 → cost-benefit matrix 구성 → 각 모델별 최적 압축 전략 및 비압축 대비 순 효율성 도출 · 이미 풀렸을 위험: 중간. P9는 모델 의존성을 보였으나, 실제 compute cost-benefit 분석은 부재. 이는 실무 relevance를 높이는 실용적 novelty. · 근거: P9
- **G3** (missing ablation) — 모르는 것: 메모리 충돌의 분류 체계(taxonomy): 1) 같은 slot vs 다른 slot의 충돌, 2) fact vs opinion vs hedged statement 충돌, 3) 최신성(recency) vs 정확성(correctness) 간 conflict. 각 충돌 유형이 에이전트 성능에 미치는 차등 영향은 무엇인가? 회피 전략(더 많은 메모리 슬롯)이 해결 전략(conflict resolution)보다 효율적인 조건은? · 검증: 1) Conflict taxonomy 정의 및 synthetic conflict 데이터셋 생성, 2) controlled conflict injection: 각 유형의 conflict를 다양한 복잡도로 agent 메모리에 주입, 3) 유형별 성능 저하 측정 → impact ranking, 4) 회피(메모리 확장) vs 해결(resolution) 전략의 cost-performance 비교 · 이미 풀렸을 위험: 중간-높음. Continual learning에서 catastrophic forgetting은 잘 알려졌으나, LLM 메모리 시스템의 conflict taxonomy와 해결 전략의 comparative effectiveness는 미개발. 이는 높은 novelty 잠재력. · 근거: P3, P5
- **G6** (evaluation weakness) — 모르는 것: 메모리 시스템 성능 평가의 신뢰도 정의: 1) 어떤 지표(accuracy, F1, recall, task completion rate, reward)가 메모리 성능을 공정하게 반영하는가? 2) LLM-as-judge vs automated metrics vs human annotation의 일관성과 신뢰도는? 3) 다양한 평가자 간(또는 모델 간) 평가 일관성(inter-rater agreement)은 충분한가? · 검증: 1) 샘플 메모리 질의에 대해 인간 주석(consensus) baseline 구축, 2) LLM-as-judge, automated metrics (exact match, F1), downstream task success rate와 인간 baseline의 correlation 측정, 3) 다양한 LLM 판정자(GPT, Claude, Llama)의 일관성 측정 → 신뢰할 수 있는 평가 방법론 정의 · 이미 풀렸을 위험: 낮음. 평가 방법론은 모든 연구에 필수이고 일반적 원칙은 알려져 있다. 하지만 메모리 시스템 특화 평가 프로토콜(예: 시간적 정확성, 한정적 정확성, 충돌 해결 정확성)은 미개발. · 근거: P2
- **G7** (missing ablation) — 모르는 것: 메모리 관리 전략(예: 경험 재생, conflict resolution, compression)의 generalization across domains. 웹 네비게이션에서 효과적인 전략이 대화형 추론(dialogue reasoning), 코딩 에이전트(code generation), 계획 수립(planning)에서도 효과적인가? 도메인 특성(task structure, memory churn rate, conflict frequency)에 따라 최적 전략이 달라지는가? · 검증: 3-4개 도메인(web navigation, dialogue reasoning, coding agents, task planning)에서 동일 메모리 전략 구현 및 평가 → domain-wise performance variation 측정 → correlation analysis로 domain characteristics와 strategy effectiveness의 관계 분석 → generalization 가능성 판단 · 이미 풀렸을 위험: 중간. 각 논문이 특정 도메인만 평가했으므로 gap은 명확하다. 하지만 기존 AI 연구에서도 도메인 전이는 최근 관심사이므로 novelty risk 중간 수준. 다만 메모리 시스템 특화 도메인 간 비교 연구는 부재할 가능성 높음. · 근거: P1, P4

## 진화 계보

- H1 → **H1-v2** (feasibility): 오라클 케이스 규모 축소: 50→15-25 (고분산 시나리오 선택으로 정보 가치 유지); 선형 보간 제거: 비용 많이 드는 full interpolation 대신 P5/P6 기반 분석적 상한 사용 (비선형 손실 환경에서의 보간 편향 제거); 독립성 가정 폐기: write-retrieval coupling을 명시적 모델로 통합 (P5 간섭, P6 시간 정보 손실로 interaction magnitude 추정)
- H1 → **H4** (out_of_box): oracle substitution 진단 방법론(H1) 제거 → 검색 품질을 직접 제어하는 경량 실험으로 피벗; write-side loss 진단 목표 제거 → P7 증거 기반 retrieval 중심으로 초점 전환; Oracle 구성 50개 + interpolation 계산 (48시간 추정) → 직접 측정 12개 configuration (36시간 추정), 계산 비용 75% 절감

## 라운드별 메타리뷰 피드백

**Round 0** — 반복 약점: Baseline fairness violation across all hypotheses: strategies compared with unequal compute budgets (H1 oracle construction, H2 model switching, H3 joint tuning) without pre-commitment to how compute parity is enforced or measured. Different budget allocation schemes (sequential vs. parallel, tuning vs. inference) produce different comparisons and enable arbitrary winner selection.; Unvalidated core assumptions driving mechanism selection: H2 assumes monotonic churn↔model-size relationship without evidence; H3 assumes integration > independent without controlling for joint-tuning confound; H1 assumes write-retrieval independence despite P5, P6 evidence of interaction. These assumptions are architectural premises, not validated findings.; Disconnect between proposed mechanism and observed bottleneck: H1 and H3 optimize write-side despite P7 evidence (retrieval dominates accuracy). This creates risk of wasted engineering effort on non-binding constraints while true bottleneck remains unaddressed.; Experimental design details under-specified until after hypothesis selection: efficiency normalization scheme (H1), heuristic thresholds (H2), joint vs. independent tuning protocol (H3) all lack pre-commitment. Risk of post-hoc metric choice or threshold tuning on test data introduces leakage and defeats falsifiability.; Oracle construction and interpolation feasibility: H1 requires ~150 oracle cases (perfect write + perfect retrieval per strategy), but RTX A5000 compute budget (~7 days) barely accommodates 48 cases. Proposed linear interpolation in non-convex memory loss landscapes introduces systematic bias and invalidates oracle diagnostic value.; Claimed improvement magnitudes lack quantitative derivation: H3 claims 25–35% improvement but prior work (P6: ~10% temporal loss, P5: conflict-induced loss in limited scenarios, P9: compression gains that decay with model size) does not sum to claimed range. Number appears arbitrary and overstates expected benefit.
  - 다음 생성 지침: Pre-commit to measurement protocol and metric normalization before designing mechanism: specify (a) how to aggregate VRAM, latency, token budget, task accuracy into a single Pareto frontier or single score, (b) oracle case selection and interpolation method, (c) baseline compute budgets (sequential vs. parallel) and parity enforcement, (d) train/validation/test split for heuristic tuning. Document in appendix; do not revise after observing results.; Validate write-side bottleneck with lightweight oracle experiments first: before proposing write-side mechanisms, run oracle substitution on representative tasks (perfect write + imperfect retrieval; imperfect write + perfect retrieval) to estimate write-loss contribution. If retrieval loss dominates (>60%), pivot to retrieval-focused hypotheses (retrieve P7 evidence: retrieval quality dominates accuracy).; Design jointly-tuned independent baselines to isolate integration benefit from tuning benefit: for H3-style integration claims, include three variants in same experiment: (a) integrated system, (b) independent mechanisms with separate tuning budgets, (c) independent mechanisms with joint compute budget but modular architecture. Only (a) vs (c) difference reflects true architectural synergy.; Reduce oracle computational burden via smaller high-value case set or analytical bounds: replace 150-case interpolation with (a) 15–25 oracle cases on high-variance scenarios (e.g., query types with worst retrieval performance), (b) analytical bounds from P5, P6 on expected write-loss magnitude, or (c) lightweight proxy metrics (e.g., memory fragmentation, token-per-update). Feasibility >> perfect oracle.; Prioritize retrieval-side hypotheses given P7 evidence: if retrieval quality dominates accuracy in agent memory systems, design H4–H6 targeting retrieval efficiency/quality before spending engineering effort on write-side. Consider adaptive retrieval (H2-style switching but for retrieval methods), cascade retrieval, or retrieval + ranking retraining with task-specific signals.; Quantitatively justify improvement claims from first-principles component analysis: if claiming K% improvement, show this derives from component gains in P3–P9 (e.g., 'temporal loss ~10% [P6] + conflict loss ~5% [P5] in this scenario + compression gains estimated at 8–12% [P9] = 23–27% total if orthogonal'). Unexplained remainder flags either new discovery (requires careful validation) or over-claimed improvement.; Model or measure write-retrieval interactions explicitly, not as nuisance: P5, P6 suggest coupling. Design experiment or metric that isolates interaction cost (e.g., 'metadata loss from write reduces subsequent retrieval by X%'). Include interaction term in hypothesis; treat as feature, not confound.

## 문헌 (검색으로 수집된 논문만)

| ID | 제목 | 연도 | 출처 |
|---|---|---|---|
| P1 | [Fixture: Long-term memory benchmark for conversational agents](https://arxiv.org/abs/9925.01000) | 2025 | fixture |
| P2 | [Fixture: LLM-as-judge leniency on memory benchmarks](https://arxiv.org/abs/9926.01010) | 2026 | fixture |
| P3 | [Fixture: Deterministic freshness resolution for memory conflicts](https://arxiv.org/abs/9926.01006) | 2026 | fixture |
| P4 | [Fixture: Experience replay memory for LLM agents in web tasks](https://arxiv.org/abs/9924.01007) | 2024 | fixture |
| P5 | [Fixture: Interference between conflicting memories in continual agents](https://arxiv.org/abs/9926.01009) | 2026 | fixture |
| P6 | [Fixture: Write-time fact extraction loses temporal cues](https://arxiv.org/abs/9926.01001) | 2026 | fixture |
| P7 | [Fixture: Retrieval dominates agent memory bottlenecks](https://arxiv.org/abs/9926.01003) | 2026 | fixture |
| P8 | [Fixture: Oracle substitution diagnosis of write and retrieval loss](https://arxiv.org/abs/9926.01004) | 2026 | fixture |
| P9 | [Fixture: Compression helps weak readers more than strong readers](https://arxiv.org/abs/9926.01002) | 2026 | fixture |
| P10 | [Fixture: Elo tournaments for ranking generated research hypotheses](https://arxiv.org/abs/9925.01005) | 2025 | fixture |
| P11 | [Fixture: Small language models as memory writers](https://arxiv.org/abs/9926.01008) | 2026 | fixture |
| P12 | [Fixture: Test-time compute scaling for scientific hypothesis generation](https://arxiv.org/abs/9925.01011) | 2025 | fixture |


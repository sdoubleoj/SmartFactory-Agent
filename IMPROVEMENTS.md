# Manufacturing AI Agent — 보완 / 개선 포인트

`main.ipynb`는 README의 설계를 LangGraph로 동작 가능한 형태로 구현한 **참조 구현(reference)** 이다.
포트폴리오 데모로는 충분하지만, 실서비스/평가까지 끌고 가려면 다음 항목들을 보완해야 한다.

---

## 1. 메모리 계층 (Memory)

### 현재 상태
- 단기 메모리: `MemorySaver` (in-memory) — 프로세스 재시작 시 thread state 소실
- 장기 메모리: 직접 만든 `SqliteLongTermMemory` (대화/요약/run trace)
- 영속 checkpoint: `SqliteSaver` 데모 셀이 있지만 기본 그래프는 `MemorySaver` 사용

### 보완해야 할 점
- **단기 메모리 영속화 일원화**: 기본 그래프부터 `SqliteSaver`(또는 `AsyncSqliteSaver`)로 통일. 데모용 `MemorySaver`는 테스트 픽스처로 분리.
- **장기 메모리 표준화**: 현재는 자체 SQLite 스키마. LangGraph의 `BaseStore`(예: `SqliteStore`)를 구현해 cross-thread namespace 검색이 가능하도록 한다. 이래야 supervisor/agent에서 `store.search(namespace, query)` 인터페이스로 통일된다.
- **세션 격리**: `thread_id == session_id` 로 단순화돼 있음. 멀티 유저 환경에서는 `(user_id, session_id)` 복합키 + RLS(권한)까지 고려해야 한다.
- **요약 압축(summary compaction)**: 현재는 raw turn을 그대로 누적. 일정 길이 초과 시 요약본으로 교체하는 compaction 작업이 없다.
- **PII/민감정보 마스킹**: 저장 전에 작업자 이름, 라인 ID 같은 민감 정보를 마스킹하는 단계가 없다.

---

## 2. Context Engineering

### 현재 상태
- 정규식 기반 센서값 추출 + 룰 기반 selector/normalizer
- prompt-injection 패턴은 5개 정도만 차단
- 토큰 예산은 단순 글자수 잘라내기

### 보완해야 할 점
- **NER/LLM 기반 추출**: 현재 regex는 표현이 바뀌면 깨진다 (예: "토크는 60Nm 정도"). 소형 LLM이나 NER 모델로 robust하게 추출.
- **단위 정규화 강화**: `°C → K`, `rpm`, `Nm` 변환 누락. 사용자가 °C로 적으면 그대로 들어간다.
- **stale 판정 정책**: 현재는 단순히 "이전값=stale". 실제로는 timestamp/턴 거리 기반으로 stale 임계값을 정해야 한다.
- **충돌 해결 로깅**: 현재는 warning 문자열만 남김. 추후 평가용으로 conflict event를 별도 테이블로 적재해야 한다.
- **prompt injection 방어**: 정규식 5개로 충분하지 않다. allow-list 기반 instruction filtering, system/user 분리, classifier 모델 도입 필요.
- **token budget 실측화**: tiktoken 등으로 실제 토큰 카운팅 후 budget 적용.

---

## 3. RAG / EvidenceAgent

### 현재 상태
- ChromaDB `PersistentClient` + DefaultEmbeddingFunction (sentence-transformers)
- seed 문서 6개 하드코딩
- profile 선택은 키워드 매칭

### 보완해야 할 점
- **임베딩 모델 선택**: DefaultEmbeddingFunction은 영문 위주. 한국어 매뉴얼이면 `bge-m3`, `multilingual-e5-large`, OpenAI `text-embedding-3-large` 등으로 교체 필요.
- **문서 ingestion 파이프라인 부재**: PDF/HTML/매뉴얼 파서 → 청크 분할 → 메타데이터 부여 → ingest 단계가 없다. `unstructured`, `langchain-text-splitters` 도입.
- **리랭킹 미구현**: README는 `reranking`을 명시했지만 구현 안 됨. cross-encoder reranker (`bge-reranker-v2-m3`) 도입.
- **adaptive query**: 현재 query fan-out은 partial_risks에서 fault 이름만 붙임. HyDE / query rewriting / sub-query decomposition을 LLM으로 수행해야 한다.
- **evidence grading**: 검색 결과의 신뢰도 점수, 충분성 판단(LLM grader)이 없다. EvidenceGate는 단순 개수 체크만 한다.
- **citation 검증**: 답변에 인용된 cite_id가 실제 검색 결과에 존재하는지 검증하는 단계 필요(hallucinated citation 방지).

---

## 4. Agents

### PredictionAgent
- ML 모델이 `LogisticRegression` baseline + StandardScaler. 클래스 불균형 대응 미흡 (class_weight=balanced 만).
- **개선**: XGBoost/LightGBM, SMOTE, calibration, threshold tuning, 부분 ML(현재는 5개 feature 전부 필요).
- 오차범위/신뢰구간 미산출. ML proba에 대한 confidence interval(예: bootstrap)이 필요.
- 모델 학습 코드가 매 노트북 실행마다 동작 → 학습/추론 분리 (joblib 캐시).

### EvidenceAgent
- 검색만 하고 답변에 인용할 핵심 문장 추출(highlight)이 없다.
- 검색 실패 시 fallback_broad 전환 자동화 안 됨 (Gate 결과를 supervisor가 보고 재라우팅하는 retry policy 미완성).

### SafetyAgent
- 단순 regex 패턴 매처. LLM 기반 risk classification 필요.
- LOTO, MSDS, 회사별 작업표준 등의 정책 문서를 별도 RAG 인덱스로 두고 evidence 기반 판단으로 강화해야 한다.

---

## 5. Gates & Supervisor

### 현재 상태
- 5개 Gate 모두 구현되었으나 status enum만 채우고 끝.
- Supervisor는 "결과 없으면 그 agent 실행" 수준의 단순 라우팅.
- retry/redirect/clarification/safe block 분기 미구현.

### 보완해야 할 점
- **retry 정책 동작화**: `RETRY_LIMITS`는 정의만 돼 있고 실제 retry 루프가 없다. Gate status를 보고 supervisor가 같은 agent를 다시 부르는 conditional edge가 필요하다.
- **clarification 분기**: PredictionGate가 `ASK_MISSING_INPUT`일 때 사용자에게 질문을 되돌리는 흐름이 없다. 현재는 final_answer로 그냥 흘러간다.
- **OutputGate REWRITE 처리**: REWRITE status가 나와도 답변을 다시 만드는 로직이 없다. final_answer → output_gate → (조건부) final_answer 루프가 필요.
- **GateReport 검색성**: list[dict]에 append만 함. gate_name별 latest를 찾는 헬퍼는 있지만 SQL/indexed lookup이 없다.

---

## 6. LangGraph 구조

### 현재 상태
- StateGraph + Conditional edges로 README 흐름 거의 그대로 구현
- TypedDict 기반 state, Pydantic은 contract validation 용도

### 보완해야 할 점
- **state merge 정책**: list 필드(`gate_reports`)는 매 노드마다 전체 리스트를 반환해 덮어쓰기 한다. `Annotated[list, operator.add]` 같은 reducer 사용이 더 안전하다.
- **streaming**: `graph.invoke`만 사용. UI 연결 시 `graph.stream(mode='updates')` 또는 `astream_events`로 토큰 단위 스트리밍을 고려.
- **interrupt 지원**: 사람의 확인이 필요한 안전 차단 케이스에서 `interrupt_before=['final_answer']` 같은 HITL 패턴이 없다.
- **observability**: LangSmith trace 연동(`LANGSMITH_TRACING=true`)이 비활성. 운영 단계에서는 필수.
- **그래프 시각화**: `graph.get_graph().draw_mermaid()` 결과를 docs로 export하는 셀이 없다.

---

## 7. LLM 통합

### 현재 상태
- 모든 노드가 **결정론적 규칙/검색**으로만 동작. 실제 LLM 호출이 없다.

### 보완해야 할 점
- FinalAnswerNode를 LLM(예: GPT-4o, Claude Sonnet)로 자연어 답변 생성하도록 교체. 현재는 템플릿 문자열 조립.
- Supervisor를 LLM 기반 router로 교체할 수도 있음 (또는 현재처럼 결정론적 router 유지). 각각 trade-off가 있으니 평가 필요.
- Safety/Evidence Agent도 LLM과 service의 hybrid 구조로 발전시켜야 답변 품질이 올라간다.

---

## 8. 평가 / 테스트

### 현재 상태
- 데모 3턴(`run_turn`) 외 자동화 테스트 없음.

### 보완해야 할 점
- **단위 테스트**: 각 service(`PredictionService.rule_based_partial`, `RagService.search`)의 pytest.
- **그래프 통합 테스트**: 시나리오별 expected route + final_answer 스냅샷.
- **회귀 케이스 모음**: prompt injection / 단위 충돌 / 누락 feature / safety bypass 등 edge case fixture.
- **품질 메트릭**: faithfulness, answer relevancy, citation precision (Ragas/Trulens 등).

---

## 9. 운영 / 배포

### 보완해야 할 점
- **노트북 → 모듈 분리**: 현재 단일 ipynb. README가 정의한 폴더 구조(`agents/`, `gates/`, ...)대로 실제 패키지로 분리.
- **설정 외부화**: DB 경로, ChromaDB 컬렉션명, 모델명 등이 셀에 하드코딩. `pydantic-settings` + `.env`로 관리.
- **컨테이너화**: ChromaDB는 docker-compose로 분리해서 별도 서비스로 운영하는 패턴이 일반적.
- **권한/감사 로그**: SafetyDecision/BLOCK 이벤트는 별도 audit log 시스템(예: append-only)으로 보내야 한다.

---

## 10. 데이터/스키마 보완

- AI4I 데이터셋만으로는 실제 매뉴얼/안전 정책이 부족. seed 6문서는 데모일 뿐 → 사내 매뉴얼/규정/QC 보고서를 별도 RAG 인덱스로 분리.
- `Type L/M/H` 의미를 일반 사용자도 알도록 prompt/answer에서 풀어 설명하는 보강 필요.
- 한국어/영어 혼용 환경 대응(컬럼명은 영문, 사용자 질문은 한국어).

---

## 우선순위 추천 (있는 자원으로 즉시 개선)

| 순위 | 항목 | 이유 |
|---|---|---|
| 1 | LangGraph `gate_reports` reducer 패턴 + retry 분기 | 현재 supervisor가 retry/clarification을 못 함 |
| 2 | FinalAnswerNode를 LLM 기반으로 교체 + citation 검증 | 가장 사용자에게 보이는 품질 차이 |
| 3 | RAG reranker + 한국어 임베딩 모델 | 검색 품질이 답변 품질의 천장 |
| 4 | `MemorySaver` → `SqliteSaver` 통일 + LangGraph `BaseStore` 구현 | 영속성/세션 격리 정합성 |
| 5 | Pytest 회귀 케이스 모음 + LangSmith trace | 이후 변경에 대한 안전망 |

# 25. Research Frontiers

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 테스트 시간 연산 확장(test-time compute scaling)을 설명하고 o1-style 추론이 확장된 연쇄 추론(chain-of-thought reasoning) 생성을 통해 복잡한 문제에서 성능을 어떻게 향상시키는지 서술할 수 있다
2. World Models 패러다임을 설명하고 Sora 같은 모델이 비디오 데이터로부터 잠재적 물리 시뮬레이션(latent physical simulation)을 어떻게 학습하는지 설명할 수 있다
3. 합성 데이터(synthetic data) 생성이 인터넷 규모 코퍼스를 넘어 파운데이션 모델 학습을 확장하는 데 어떤 역할을 하는지 분석할 수 있다
4. 멀티 에이전트(multi-agent) 시스템 아키텍처를 비교하고 특화된 에이전트들이 협업하여 단일 모델의 역량을 넘어서는 태스크를 해결하는 방법을 설명할 수 있다
5. 파운데이션 모델의 새로운 연구 방향을 평가하고 향후 수 년간 AI 역량에 미칠 잠재적 영향을 가늠할 수 있다

---

## 개요

Foundation Model 연구의 최전선을 다룹니다. World Models, o1-style Reasoning, Synthetic Data, Multi-Agent 시스템 등 미래 방향을 탐구합니다.

---

## 1. o1-style Reasoning (Test-time Compute)

### 1.1 개념

```
기존 LLM vs o1-style:
┌─────────────────────────────────────────────────────────┐
│  기존 LLM:                                              │
│  - 학습 시간에 계산 집중 (더 큰 모델, 더 많은 데이터)   │
│  - 추론 시 고정된 forward pass                          │
│  - 복잡한 문제에 한계                                   │
│                                                         │
│  o1-style (Test-time Compute Scaling):                 │
│  - 추론 시 더 많은 계산 사용                            │
│  - Chain-of-Thought 자동 생성                          │
│  - 여러 경로 탐색 후 최선 선택                          │
│  - 문제 난이도에 따라 적응적 계산                       │
└─────────────────────────────────────────────────────────┘

핵심 기법:
1. Internal Chain-of-Thought
2. Search/Verification loops
3. Self-consistency checking
4. Reward model guided search
```

### 1.2 개념적 구현

```python
import torch
from typing import List, Tuple

class ReasoningModel:
    """o1-style 추론 모델 (개념적 구현)"""

    def __init__(self, base_model, reward_model):
        self.model = base_model
        self.reward_model = reward_model

    def reason(
        self,
        problem: str,
        max_thinking_tokens: int = 10000,
        num_candidates: int = 5
    ) -> str:
        """확장된 추론"""
        # 1. 여러 reasoning chain 생성
        candidates = self._generate_candidates(problem, num_candidates)

        # 2. 각 chain 평가
        scored_candidates = []
        for chain, answer in candidates:
            score = self._evaluate_chain(chain, answer)
            scored_candidates.append((chain, answer, score))

        # 3. 최선의 답변 선택
        best = max(scored_candidates, key=lambda x: x[2])
        return best[1]  # 답변만 반환 (chain은 내부)

    def _generate_candidates(
        self,
        problem: str,
        n: int
    ) -> List[Tuple[str, str]]:
        """여러 추론 경로 생성"""
        candidates = []

        for _ in range(n):
            # Step-by-step reasoning 생성
            chain = self._generate_reasoning_chain(problem)

            # Chain에서 최종 답변 추출
            answer = self._extract_answer(chain)

            candidates.append((chain, answer))

        return candidates

    def _generate_reasoning_chain(self, problem: str) -> str:
        """추론 체인 생성"""
        prompt = f"""Solve this problem step by step.
Think carefully and show your reasoning.

Problem: {problem}

Let me think through this carefully..."""

        # 길이 제한 없이 생성 (또는 매우 긴 제한)
        response = self.model.generate(
            prompt,
            max_new_tokens=5000,
            temperature=0.7
        )

        return response

    def _evaluate_chain(self, chain: str, answer: str) -> float:
        """추론 체인 품질 평가"""
        # Reward model로 평가
        score = self.reward_model.evaluate(chain)

        # 자기 일관성 체크
        consistency_score = self._check_consistency(chain, answer)

        return score * 0.7 + consistency_score * 0.3

    def _check_consistency(self, chain: str, answer: str) -> float:
        """논리적 일관성 검사"""
        # 간단한 휴리스틱 또는 별도 모델 사용
        prompt = f"""Is this reasoning chain logically consistent?

Reasoning:
{chain}

Answer: {answer}

Rate consistency (0-1):"""

        response = self.model.generate(prompt, max_new_tokens=10)
        # 파싱...
        return 0.8  # 예시


class TreeOfThoughts:
    """Tree of Thoughts 구현"""

    def __init__(self, model, evaluator):
        self.model = model
        self.evaluator = evaluator

    def solve(
        self,
        problem: str,
        depth: int = 3,
        branching_factor: int = 3
    ) -> str:
        """트리 탐색으로 문제 해결"""
        root = {"state": problem, "thoughts": [], "score": 0}
        best_path = self._search(root, depth, branching_factor)
        return self._extract_solution(best_path)

    def _search(self, node: dict, depth: int, bf: int) -> List[dict]:
        """BFS/DFS 탐색"""
        if depth == 0:
            return [node]

        # 다음 단계 생각들 생성
        thoughts = self._generate_thoughts(node, bf)

        # 각 생각 평가
        children = []
        for thought in thoughts:
            child = {
                "state": node["state"],
                "thoughts": node["thoughts"] + [thought],
                "score": self._evaluate_thought(thought, node)
            }
            children.append(child)

        # 상위 b개만 확장 (beam search)
        children.sort(key=lambda x: x["score"], reverse=True)
        children = children[:bf]

        # 재귀 탐색
        best_paths = []
        for child in children:
            path = self._search(child, depth - 1, bf)
            best_paths.extend(path)

        return sorted(best_paths, key=lambda x: x["score"], reverse=True)[:1]

    def _generate_thoughts(self, node: dict, n: int) -> List[str]:
        """다음 단계 생각 생성"""
        context = "\n".join(node["thoughts"])

        prompt = f"""Problem: {node["state"]}

Previous thoughts:
{context}

Generate {n} different next steps or approaches:"""

        response = self.model.generate(prompt)
        # 파싱하여 n개 생각 추출
        return response.split("\n")[:n]

    def _evaluate_thought(self, thought: str, node: dict) -> float:
        """생각의 품질 평가"""
        return self.evaluator.score(thought, node["state"])
```

---

## 2. Synthetic Data

### 2.1 개념

```
Synthetic Data Generation:
┌─────────────────────────────────────────────────────────┐
│  문제: 고품질 학습 데이터 부족                           │
│                                                         │
│  해결: LLM으로 학습 데이터 생성                         │
│                                                         │
│  방법:                                                  │
│  1. Self-Instruct: instruction/response 쌍 생성        │
│  2. Evol-Instruct: 점진적 복잡화                        │
│  3. Rejection Sampling: 다수 생성 후 필터링             │
│  4. RLHF-style: 선호도 데이터 생성                      │
│  5. Distillation: 강한 모델에서 약한 모델로             │
│                                                         │
│  주의:                                                  │
│  - Model collapse (자기 데이터로만 학습 시)             │
│  - 다양성 유지 중요                                     │
│  - 품질 검증 필수                                       │
└─────────────────────────────────────────────────────────┘
```

### 2.2 구현

```python
class SyntheticDataGenerator:
    """합성 데이터 생성기"""

    def __init__(self, teacher_model, student_model=None):
        self.teacher = teacher_model
        self.student = student_model

    def generate_instruction_data(
        self,
        seed_instructions: List[str],
        num_samples: int = 10000,
        diversity_threshold: float = 0.7
    ) -> List[dict]:
        """Instruction-Response 데이터 생성"""
        generated = []
        instruction_embeddings = []

        while len(generated) < num_samples:
            # 새 instruction 생성
            instruction = self._generate_instruction(seed_instructions + [
                g["instruction"] for g in generated[-10:]
            ])

            # 다양성 체크
            if self._check_diversity(instruction, instruction_embeddings, diversity_threshold):
                # Response 생성
                response = self._generate_response(instruction)

                # 품질 체크
                if self._quality_check(instruction, response):
                    generated.append({
                        "instruction": instruction,
                        "response": response
                    })

                    # 임베딩 저장
                    emb = self._get_embedding(instruction)
                    instruction_embeddings.append(emb)

            if len(generated) % 100 == 0:
                print(f"Generated {len(generated)}/{num_samples}")

        return generated

    def _generate_instruction(self, examples: List[str]) -> str:
        """새 instruction 생성"""
        examples_text = "\n".join([f"- {ex}" for ex in examples[-5:]])

        prompt = f"""Here are some example instructions:
{examples_text}

Generate a new, different instruction that is:
1. Clear and specific
2. Different from the examples
3. Useful and educational

New instruction:"""

        return self.teacher.generate(prompt, temperature=0.9)

    def _generate_response(self, instruction: str) -> str:
        """Response 생성"""
        prompt = f"""Instruction: {instruction}

Please provide a helpful, accurate, and detailed response:"""

        return self.teacher.generate(prompt, temperature=0.7)

    def _check_diversity(
        self,
        instruction: str,
        existing_embeddings: List,
        threshold: float
    ) -> bool:
        """다양성 검사"""
        if not existing_embeddings:
            return True

        new_emb = self._get_embedding(instruction)

        for emb in existing_embeddings:
            similarity = self._cosine_similarity(new_emb, emb)
            if similarity > threshold:
                return False

        return True

    def _quality_check(self, instruction: str, response: str) -> bool:
        """품질 검사"""
        # 길이 체크
        if len(response) < 50:
            return False

        # 관련성 체크 (간단한 휴리스틱)
        instruction_words = set(instruction.lower().split())
        response_words = set(response.lower().split())

        overlap = len(instruction_words & response_words)
        if overlap < 2:
            return False

        return True


class RejectSampling:
    """Rejection Sampling으로 고품질 데이터 선별"""

    def __init__(self, generator_model, reward_model):
        self.generator = generator_model
        self.reward = reward_model

    def generate_with_rejection(
        self,
        prompt: str,
        n_samples: int = 16,
        top_k: int = 1
    ) -> List[str]:
        """다수 생성 후 최선 선택"""
        # 여러 응답 생성
        responses = []
        for _ in range(n_samples):
            response = self.generator.generate(prompt, temperature=0.8)
            responses.append(response)

        # 각 응답 점수화
        scored = []
        for response in responses:
            score = self.reward.score(prompt, response)
            scored.append((response, score))

        # 상위 k개 선택
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, s in scored[:top_k]]
```

---

## 3. Multi-Agent Systems

### 3.1 개념

```
Multi-Agent LLM Systems:
┌─────────────────────────────────────────────────────────┐
│  Agent 유형:                                            │
│                                                         │
│  1. Debate: 여러 Agent가 토론                           │
│     - 서로 다른 관점 제시                               │
│     - 합의 도출                                         │
│                                                         │
│  2. Collaboration: 역할 분담 협업                       │
│     - 작성자, 검토자, 편집자                            │
│     - 연구자, 개발자, 테스터                            │
│                                                         │
│  3. Competition: 경쟁적 생성                            │
│     - 최선의 결과 선택                                  │
│     - Red team / Blue team                              │
│                                                         │
│  4. Hierarchical: 계층적 구조                           │
│     - Manager → Worker agents                           │
│     - 태스크 분해 및 위임                               │
└─────────────────────────────────────────────────────────┘
```

### 3.2 구현

```python
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    WRITER = "writer"
    CRITIC = "critic"
    EDITOR = "editor"

@dataclass
class Message:
    sender: str
    receiver: str
    content: str
    metadata: Dict[str, Any] = None

class MultiAgentSystem:
    """다중 에이전트 시스템"""

    def __init__(self, llm):
        self.llm = llm
        self.agents = {}
        self.message_history = []

    def add_agent(self, name: str, role: AgentRole, system_prompt: str):
        """에이전트 추가"""
        self.agents[name] = {
            "role": role,
            "system_prompt": system_prompt,
            "memory": []
        }

    def send_message(self, sender: str, receiver: str, content: str):
        """메시지 전송"""
        message = Message(sender=sender, receiver=receiver, content=content)
        self.message_history.append(message)
        self.agents[receiver]["memory"].append(message)

        return self._get_response(receiver)

    def _get_response(self, agent_name: str) -> str:
        """에이전트 응답 생성"""
        agent = self.agents[agent_name]

        # 최근 메시지로 컨텍스트 구성
        recent_messages = agent["memory"][-5:]
        context = "\n".join([
            f"{m.sender}: {m.content}" for m in recent_messages
        ])

        prompt = f"""{agent["system_prompt"]}

Recent conversation:
{context}

Your response as {agent_name}:"""

        return self.llm.generate(prompt)

    def run_debate(
        self,
        topic: str,
        agents: List[str],
        rounds: int = 3
    ) -> str:
        """토론 실행"""
        # 초기 의견
        opinions = {}
        for agent in agents:
            response = self.send_message(
                "moderator", agent,
                f"What is your position on: {topic}"
            )
            opinions[agent] = response

        # 토론 라운드
        for round in range(rounds):
            for agent in agents:
                # 다른 에이전트 의견 전달
                other_opinions = "\n".join([
                    f"{a}: {o}" for a, o in opinions.items() if a != agent
                ])

                response = self.send_message(
                    "moderator", agent,
                    f"Others' opinions:\n{other_opinions}\n\nYour response:"
                )
                opinions[agent] = response

        # 합의 도출
        final_opinions = "\n".join([f"{a}: {o}" for a, o in opinions.items()])
        consensus = self.llm.generate(
            f"Based on this debate, summarize the consensus:\n{final_opinions}"
        )

        return consensus


class CollaborativeWriting:
    """협업 글쓰기 시스템"""

    def __init__(self, llm):
        self.system = MultiAgentSystem(llm)

        # 에이전트 설정
        self.system.add_agent(
            "writer",
            AgentRole.WRITER,
            "You are a creative writer. Write engaging content."
        )
        self.system.add_agent(
            "critic",
            AgentRole.CRITIC,
            "You are a critical reviewer. Point out issues and suggest improvements."
        )
        self.system.add_agent(
            "editor",
            AgentRole.EDITOR,
            "You are an editor. Refine and polish the writing."
        )

    def write(self, topic: str, iterations: int = 3) -> str:
        """협업 글쓰기"""
        # 초안 작성
        draft = self.system.send_message(
            "user", "writer",
            f"Write a short article about: {topic}"
        )

        for i in range(iterations):
            # 비평
            critique = self.system.send_message(
                "writer", "critic",
                f"Please review this draft:\n{draft}"
            )

            # 수정
            revised = self.system.send_message(
                "critic", "writer",
                f"Based on this feedback:\n{critique}\n\nPlease revise the draft."
            )

            draft = revised

        # 최종 편집
        final = self.system.send_message(
            "writer", "editor",
            f"Please polish this final draft:\n{draft}"
        )

        return final
```

---

## 4. World Models

### 4.1 개념

```
World Models:
┌─────────────────────────────────────────────────────────┐
│  목표: LLM이 세계의 동작 방식을 이해하고 시뮬레이션     │
│                                                         │
│  응용:                                                  │
│  1. Planning: 행동의 결과 예측                          │
│  2. Reasoning: 인과 관계 추론                           │
│  3. Simulation: 가상 환경 시뮬레이션                    │
│  4. Embodied AI: 로봇 제어                              │
│                                                         │
│  연구 방향:                                             │
│  - Video generation as world simulation (Sora)          │
│  - Physical reasoning benchmarks                        │
│  - Embodied language models                             │
│  - Causal reasoning                                     │
└─────────────────────────────────────────────────────────┘
```

### 4.2 개념적 구현

```python
class WorldModel:
    """World Model 개념적 구현"""

    def __init__(self, llm):
        self.llm = llm
        self.state = {}

    def initialize_state(self, description: str):
        """초기 상태 설정"""
        prompt = f"""Parse this scene description into structured state.

Description: {description}

Extract:
- Objects (name, position, properties)
- Relationships between objects
- Physical constraints

State:"""

        state_text = self.llm.generate(prompt)
        self.state = self._parse_state(state_text)

    def predict_action_result(self, action: str) -> Dict:
        """행동 결과 예측"""
        state_description = self._describe_state()

        prompt = f"""Current state:
{state_description}

Action: {action}

Predict:
1. What changes will occur?
2. What is the new state?
3. Any unexpected effects?

Prediction:"""

        prediction = self.llm.generate(prompt)
        return self._parse_prediction(prediction)

    def simulate_sequence(
        self,
        actions: List[str]
    ) -> List[Dict]:
        """행동 시퀀스 시뮬레이션"""
        states = [self.state.copy()]

        for action in actions:
            prediction = self.predict_action_result(action)
            self._apply_changes(prediction)
            states.append(self.state.copy())

        return states

    def _describe_state(self) -> str:
        """상태를 텍스트로 설명"""
        # state dict를 자연어로 변환
        return str(self.state)

    def _parse_state(self, text: str) -> Dict:
        """텍스트를 상태로 파싱"""
        # 실제로는 더 정교한 파싱 필요
        return {"raw": text}

    def _parse_prediction(self, text: str) -> Dict:
        """예측 결과 파싱"""
        return {"raw": text}

    def _apply_changes(self, prediction: Dict):
        """예측된 변화 적용"""
        # 상태 업데이트
        pass
```

---

## 5. 미래 연구 방향

### 5.1 주요 방향

```
1. Scaling Laws Beyond Parameters
   - Test-time compute scaling
   - Mixture of Experts scaling
   - Data quality over quantity

2. Multimodal Understanding
   - Native multimodal models
   - Embodied AI
   - Physical world understanding

3. Reasoning Enhancement
   - Formal verification
   - Neuro-symbolic integration
   - Causal reasoning

4. Alignment & Safety
   - Constitutional AI
   - Interpretability
   - Robustness to adversarial inputs

5. Efficiency
   - Sparse architectures
   - Mixture of Depths
   - Early exit mechanisms
```

### 5.2 열린 문제들

```
┌─────────────────────────────────────────────────────────┐
│  열린 연구 문제:                                        │
│                                                         │
│  1. Hallucination 완전 해결                             │
│     - 언제 모르는지 아는 것                             │
│     - 신뢰도 calibration                                │
│                                                         │
│  2. True Reasoning vs Pattern Matching                  │
│     - 진정한 일반화 능력?                               │
│     - Out-of-distribution 추론                          │
│                                                         │
│  3. Long-term Memory                                    │
│     - 영구적 학습                                       │
│     - Continual learning without forgetting             │
│                                                         │
│  4. Efficiency-Capability Tradeoff                      │
│     - 작은 모델의 한계?                                 │
│     - Knowledge distillation 한계                       │
│                                                         │
│  5. Alignment                                           │
│     - Value alignment의 정의                            │
│     - Scalable oversight                                │
└─────────────────────────────────────────────────────────┘
```

---

## 핵심 정리

### Research Frontiers 요약
```
1. o1-style: 추론 시 더 많은 계산
2. Synthetic Data: LLM으로 학습 데이터 생성
3. Multi-Agent: 협업/토론/경쟁 시스템
4. World Models: 물리 세계 시뮬레이션
5. Alignment: 안전하고 유용한 AI
```

### 미래 전망
```
- Parameter scaling → Compute scaling
- Single model → Multi-agent systems
- Text → Native multimodal
- Pattern matching → True reasoning
- Black box → Interpretable
```

---

## 참고 자료

1. OpenAI (2024). "Learning to Reason with LLMs" (o1)
2. Yao et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
3. Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior"
4. Ha & Schmidhuber (2018). "World Models"

---

## 연습 문제

### 연습 문제 1: 테스트 시간 컴퓨팅(Test-Time Compute) 스케일링 트레이드오프
o1-스타일 모델은 훈련 시간 스케일링에만 의존하는 대신 추론 중에 더 많은 컴퓨팅을 사용합니다. 다음 각 시나리오에 대해 두 가지 패러다임을 비교하고 어느 것이 더 적합한지 설명하세요.

| 시나리오 | 훈련 시간 스케일링 | 테스트 시간 컴퓨팅 스케일링 | 더 나은 선택 | 이유 |
|----------|----------------------|--------------------------|---------------|--------|
| A) 하루 100만 건 고객 지원 쿼리 처리 | ??? | ??? | ??? | ??? |
| B) 새로운 수학 경시대회 문제 풀기 | ??? | ??? | ??? | ??? |
| C) 실시간 코딩 자동 완성 (<50ms 지연) | ??? | ??? | ??? | ??? |

<details>
<summary>정답 보기</summary>

| 시나리오 | 훈련 시간 스케일링 | 테스트 시간 컴퓨팅 | 더 나은 선택 | 이유 |
|----------|----------------------|-------------------|---------------|--------|
| A) 하루 100만 건 고객 지원 | 더 큰 모델이 더 많은 쿼리 유형 처리; 고정 지연 | 어려운 쿼리에서 더 높은 품질이지만 쿼리당 지연 및 비용 증가 | **훈련 시간 스케일링** | 고객 지원은 대용량이고 예측 가능한 쿼리 유형을 가집니다. 각 쿼리는 빠르게 처리되어야 합니다. 훈련 시간 스케일링(더 크거나 더 잘 훈련된 모델)은 고정 한계 비용으로 쿼리당 일관된 품질을 제공합니다. 테스트 시간 컴퓨팅은 쿼리당 비용을 10-50배 늘릴 것입니다 — 하루 100만 건에서는 감당할 수 없습니다. |
| B) 새로운 수학 경시 문제 | 더 나은 일반 수학 지식 | 여러 해결 경로 생성, 각각 검증, 오류 시 역추적 | **테스트 시간 컴퓨팅 스케일링** | 새로운 경시 문제는 종종 올바른 방법을 찾기 전에 여러 접근법을 탐색해야 합니다. 수학자도 IMO 문제를 2초 안에 풀지 않습니다 — 몇 시간을 생각합니다. 테스트 시간 컴퓨팅이 이를 모방합니다: 어려운 문제에 10-100배 더 많은 토큰을 사용합니다. 볼륨이 낮아(문제 1개) 문제당 높은 비용이 허용됩니다. |
| C) 실시간 코드 자동 완성 (<50ms) | 모델이 추론 시 일반적인 코드 패턴을 알아야 함 | 더 많은 사고 → 더 많은 지연 → 50ms 요구사항 위반 | **훈련 시간 스케일링** | 50ms의 엄격한 지연 제약으로 테스트 시간 컴퓨팅 스케일링은 불가능합니다(추가 순전파 패스가 많이 필요). 대신: 훈련 데이터에서 일반적인 코드 패턴에 대한 높은 정확도를 가진 소형 모델(< 10억 파라미터)을 사용합니다. 추측적 디코딩(speculative decoding) 같은 기술이 속도를 향상시킬 수 있지만 사고 예산은 거의 0이어야 합니다. |

**핵심 통찰**: 테스트 시간 컴퓨팅은 **희귀하고, 어렵고, 개별 인스턴스에 더 많은 시간을 허용하는** 문제에서 가장 유용합니다. 훈련 시간 스케일링은 **대용량, 지연에 민감한, 또는 비용 제한이 있는** 애플리케이션에 더 적합합니다.

</details>

### 연습 문제 2: 합성 데이터 모델 붕괴(Model Collapse) 위험
`SyntheticDataGenerator`는 교사 모델이 생성한 데이터로 훈련됩니다. "모델 붕괴(model collapse)" 현상을 설명하고 이를 감지하고 방지하는 두 가지 구체적인 전략을 서술하세요.

```python
# 잠재적 붕괴 시나리오:
# Round 1: Train on human data → Model v1
# Round 2: Generate synthetic data with v1 → Train → Model v2
# Round 3: Generate synthetic data with v2 → Train → Model v3
# ...
# Round N: What happens to the distribution?
```

<details>
<summary>정답 보기</summary>

**모델 붕괴 메커니즘**:

각 라운드에서 모델은 자신의 분포에서 합성 훈련 데이터를 생성합니다. 각 생성 단계는 미묘한 편향을 도입하고 꼬리 분포 예제(희귀하지만 중요한 패턴)를 잃어버립니다. 다음 모델이 이 좁아진 분포로 훈련되면 편향이 더 증폭됩니다.

```
라운드 1 (인간 데이터): 분포가 {일반, 비일반, 희귀} 콘텐츠를 포함
라운드 2 (v1 합성): 일부 "희귀" 패턴 제거 (낮은 확률 → 드물게 샘플링)
라운드 3 (v2 합성): 분포가 더 축소, v1의 편향 증폭
라운드 N: 분포가 고확률, 반복적인 출력으로 붕괴
         - 어휘 다양성 감소
         - 반복적인 문체 패턴
         - 사실적 엣지 케이스 손실
         - 불확실한 질문에 과도한 자신감 (v1의 확신에 찬 틀린 답변이 전파됨)
```

**방지 전략 1: 인간 데이터 리플레이(Replay)**

```python
def train_with_replay(synthetic_data, human_data, replay_ratio=0.1):
    """Always include a fraction of original human data"""
    # In every training batch: 10% human data + 90% synthetic
    # Human data anchors the distribution to the real world
    # Prevents drift toward narrow synthetic distribution
    human_sample = random.sample(human_data, int(len(synthetic_data) * replay_ratio))
    combined = synthetic_data + human_sample
    return combined
```
모델이 항상 "실제" 인간 데이터를 보도록 하여 분포가 완전히 붕괴되는 것을 방지합니다.

**방지 전략 2: 분포 발산(Divergence) 모니터링**

```python
def check_distribution_health(model_v_current, model_v_previous, test_prompts):
    """Detect collapse by comparing output distributions"""
    # Generate outputs from both model versions
    outputs_current = [model_v_current.generate(p) for p in test_prompts]
    outputs_previous = [model_v_previous.generate(p) for p in test_prompts]

    # Measure: vocabulary diversity (unique tokens / total tokens)
    diversity_current = len(set(tokenize(outputs_current))) / len(tokenize(outputs_current))
    diversity_previous = len(set(tokenize(outputs_previous))) / len(tokenize(outputs_previous))

    # Alert if diversity drops > 10% between generations
    if diversity_current < diversity_previous * 0.9:
        raise CollapseWarning(f"Distribution shrinking: {diversity_previous:.3f} → {diversity_current:.3f}")

    # Also monitor: perplexity on human-written held-out text
    # Increasing perplexity = model drifting from human distribution
```

</details>

### 연습 문제 3: 다중 에이전트 토론(Multi-Agent Debate) vs. 단일 모델
다중 에이전트 토론 시스템에서 여러 LLM이 서로 다른 입장을 주장한 다음 합의에 도달합니다. 토론이 답변 품질을 향상시키는 문제 유형과 잠재적으로 저하시키는 유형을 구분하세요.

다음 태스크를 평가하세요:
- 태스크 A: "다음 파이썬 코드가 올바른가요? [미묘한 off-by-one 오류가 있는 함수]"
- 태스크 B: "프랑스의 수도는 어디인가요?"
- 태스크 C: "회사가 이익과 직원 복지 중 무엇을 우선시해야 하는가?"

<details>
<summary>정답 보기</summary>

**태스크 A: 미묘한 버그가 있는 코드 정확도 — 토론이 도움됨**

서로 다른 에이전트가 독립적으로 코드를 검토하며 다양한 각도에서 off-by-one 오류를 발견할 수 있습니다:
- 에이전트 1: "루프 조건이 `<`를 사용하지만 다음 이유로 `<=`여야 합니다..."
- 에이전트 2: "`i=1`에서 초기화가 첫 번째 요소를 놓칩니다..."
- 에이전트 3: "배열 길이가 0이면 충돌합니다 왜냐하면..."

적대적 구조가 각 에이전트를 결함을 찾도록 강제합니다. 단일 모델은 "올바르게 보입니다"(프롬프트의 암묵적 가정에 대한 아첨)라고 확신할 수 있습니다. 독립적인 검토를 하는 다중 에이전트는 버그를 더 잘 발견합니다.

**태스크 B: 프랑스의 수도 — 토론이 해롭거나 (자원 낭비)**

"파리"는 모호함이 없는 사실적 답변입니다. 에이전트가 토론하면:
- 에이전트 1: "파리"
- 에이전트 2: "파리"
- 에이전트 3: "파리, 하지만 역사적으로 루이 14세 시대에는 베르사유가 수도 역할을 했습니다..."

토론은 지연, 비용을 추가하며 최종 답변을 혼란스럽게 만드는 불필요한 "흥미로운" 한정어를 에이전트가 도입할 가능성이 있습니다. 결정론적 답변이 있는 사실적이고 폐쇄형 질문에는 단일 모델이 더 낫습니다.

**태스크 C: 이익 vs. 직원 복지 — 토론이 도움됨 (하지만 양극화 위험)**

이것은 여러 유효한 관점이 있는 진정으로 논쟁적인 질문입니다. 토론 구조는 다음을 드러낼 수 있습니다:
- 경제적 효율성 논거 (이익이 재투자를 가능하게 함 → 장기적 직원 이익)
- 사회적 계약 논거 (직원은 자원이 아닌 이해관계자)
- 경험적 증거 (높은 직원 만족도를 가진 기업이 장기적으로 성과가 좋음)

**위험**: 에이전트가 고정된 "입장"에 배정되면(에이전트 A = 이익 찬성, 에이전트 B = 직원 찬성), 미묘한 합성에 도달하기보다 양극화를 증폭시키는 설득력 있지만 일방적인 논거를 생성할 수 있습니다. 고정 배정 없이 에이전트가 여러 독립적 관점을 생성하도록 하는 것이 더 좋습니다.

**일반 규칙**: 토론은 **추론 문제**(여러 경로를 교차 확인할 수 있는), **적대적 태스크**(버그 찾기, 보안 검토), **가치 관련 질문**(여러 관점이 뉘앙스를 추가하는)에서 품질을 향상시킵니다. **사실 조회**, **단순 계산**, **모든 모델이 어차피 동의하는 합의 태스크**에서는 비용만 추가합니다.

</details>

### 연습 문제 4: 물리적 추론을 위한 세계 모델(World Models)
Sora 스타일의 세계 모델은 비디오 데이터에서 물리적 환경을 시뮬레이션하는 법을 학습합니다. 현재 세계 모델이 신뢰할 수 있는 물리 시뮬레이터로 사용되지 못하게 하는 근본적인 한계 하나를 설명하고, 이를 해결하기 위해 어떤 종류의 훈련 데이터가 필요한지 서술하세요.

<details>
<summary>정답 보기</summary>

**근본적인 한계: 인과적 근거 부재 (상관관계 vs. 인과관계)**

현재 비디오 기반 세계 모델은 픽셀 공간에서 **통계적 상관관계**를 학습합니다: "이 프레임 구성을 보면, 다음 프레임은 보통 이렇게 됩니다." 그들은 그 상관관계를 생성하는 **인과적 물리 법칙**(뉴턴의 법칙, 에너지 보존, 유체 역학 방정식)을 학습하지 않습니다.

**구체적인 예시**:
경사면을 굴러 내려가는 공 비디오로 훈련된 세계 모델은 "경사면 위의 공 → 굴러가는 공 → 경사면 아래의 공"을 예측하는 법을 학습합니다. 하지만 이는 힘 = 질량 × 가속도를 모델링하는 것이 아니라 시각적 궤적 패턴을 암기함으로써 이루어집니다.

**실패 모드**:
- **반사실적 추론**: "공이 두 배 무거우면 어떻게 굴러갈까?" — 모델이 시각적 패턴만 학습했을 뿐 질량-가속도 관계를 학습하지 않았으므로 답할 수 없습니다.
- **새로운 물리 구성**: 훈련 비디오에서 본 적 없는 형태의 경사면은 물리적으로 불가능한 궤적을 생성할 수 있습니다 (공이 경사면 표면을 통과함).
- **장기 외삽**: 단기 예측이 정확하더라도(비디오에서 학습), 10초 앞을 외삽하면 학습된 분포에서 작은 편차가 누적되어 오류가 증폭됩니다.

**이를 해결하기 위해 필요한 훈련 데이터**:

1. **레이블이 붙은 물리 시뮬레이션** (예: 게임 엔진): 각 프레임에서 실제 물리적 양(속도, 힘, 에너지)과 쌍을 이루는 비디오. 모델이 픽셀 값만이 아니라 이러한 양을 예측하는 법을 학습할 수 있습니다.

2. **개입적 데이터(Interventional Data)** (반사실적 쌍): 다른 물체 질량, 마찰 계수 등으로 구성된 동일한 물리적 설정. "공 A(1kg)가 2초에 위치 X까지 굴러감; 공 B(2kg)가 2초에 위치 X까지 굴러감" — 이는 경사면 궤적의 질량 독립성을 가르쳐 모델을 실제 인과 구조에 근거를 두게 합니다.

3. **장기 물리 벤치마크**: 30초 이상의 외삽을 테스트하는 평가 데이터셋, 모델이 단기 패턴 매칭이 아닌 안정적인 물리적 표현을 학습하도록 강제합니다.

핵심 통찰: 시각적 세계 모델은 **예측** 태스크(다음에 무슨 일이 일어날까?)에는 훌륭하지만 **인과적** 태스크(왜 일어났는가, 다르면 어떻게 될까?)에는 불충분합니다.

</details>
5. Sora Technical Report (2024)

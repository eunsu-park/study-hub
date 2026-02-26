# 컴퓨터 그래픽스 (Computer Graphics)

## 개요

컴퓨터 그래픽스는 컴퓨터를 사용하여 이미지를 생성하는 과학이자 예술입니다 — 화면의 2D 도형부터 사실적인 3D 렌더링과 실시간 인터랙티브 세계까지. 이 토픽은 수학적 기초(변환, 투영, 셰이딩 모델), 그래픽스 파이프라인(래스터화, 텍스처링, 조명), 현대 GPU 프로그래밍(셰이더, 컴퓨트), 고급 렌더링 기법(레이 트레이싱, 전역 조명)을 다룹니다. 게임, 과학적 시각화, 영화 VFX를 제작하든, 컴퓨터 그래픽스를 이해하면 디지털 세계가 어떻게 만들어지고 표시되는지 깊이 있게 제어할 수 있습니다.

## 선수 과목

- **물리수학(Mathematical Methods)**: 선형대수, 행렬 연산 (Mathematical_Methods L03-L05)
- **프로그래밍(Programming)**: 객체지향 프로그래밍, 기본 자료구조 (Programming L05-L06)
- **Python**: NumPy, Matplotlib (Python L01-L08)
- **웹 개발(Web Development)** (선택): WebGL 레슨을 위한 HTML/CSS/JS (Web_Development L01-L08)

## 학습 경로

```
기초 (L01-L04)
├── L01: 그래픽스 파이프라인 개요
├── L02: 2D 변환
├── L03: 3D 변환과 투영
└── L04: 래스터화

셰이딩과 텍스처 (L05-L08)
├── L05: 셰이딩 모델
├── L06: 텍스처 매핑
├── L07: WebGL 기초
└── L08: 셰이더 프로그래밍 (GLSL)

고급 렌더링 (L09-L12)
├── L09: 씬 그래프와 공간 자료구조
├── L10: 레이 트레이싱 기초
├── L11: 패스 트레이싱과 전역 조명
└── L12: 애니메이션과 골격 시스템

현대 기법 (L13-L16)
├── L13: 파티클 시스템과 이펙트
├── L14: GPU 컴퓨팅
├── L15: 실시간 렌더링 기법
└── L16: 현대 그래픽스 API 개요
```

## 레슨 목록

| # | 레슨 | 설명 |
|---|------|------|
| 01 | [그래픽스 파이프라인 개요](01_Graphics_Pipeline_Overview.md) | CPU vs GPU 단계, 정점/프래그먼트 처리, 더블 버퍼링 |
| 02 | [2D 변환](02_2D_Transformations.md) | 이동, 회전, 크기 조절, 동차 좌표, 합성 |
| 03 | [3D 변환과 투영](03_3D_Transformations_and_Projections.md) | 모델/뷰/투영 행렬, 원근 vs 직교, 카메라 모델 |
| 04 | [래스터화](04_Rasterization.md) | 선 그리기, 삼각형 래스터화, Z-버퍼, 안티앨리어싱 |
| 05 | [셰이딩 모델](05_Shading_Models.md) | 퐁, 블린-퐁, PBR (Cook-Torrance), BRDF, 프레넬 |
| 06 | [텍스처 매핑](06_Texture_Mapping.md) | UV 좌표, 밉맵, 필터링, 노멀/범프 매핑, PBR 텍스처 |
| 07 | [WebGL 기초](07_WebGL_Fundamentals.md) | WebGL 컨텍스트, 버퍼, 셰이더, 드로잉, 상호작용 |
| 08 | [셰이더 프로그래밍 (GLSL)](08_Shader_Programming_GLSL.md) | 정점/프래그먼트 셰이더, 유니폼, 베어링, 내장 함수 |
| 09 | [씬 그래프와 공간 자료구조](09_Scene_Graphs_and_Spatial_Data_Structures.md) | 씬 계층 구조, BVH, 옥트리, BSP 트리, 절두체 컬링 |
| 10 | [레이 트레이싱 기초](10_Ray_Tracing_Basics.md) | 광선 생성, 교차 검사, 재귀적 레이 트레이싱, 그림자, 반사 |
| 11 | [패스 트레이싱과 전역 조명](11_Path_Tracing_and_Global_Illumination.md) | 몬테카를로 적분, 렌더링 방정식, 중요도 샘플링, 디노이징 |
| 12 | [애니메이션과 골격 시스템](12_Animation_and_Skeletal_Systems.md) | 키프레임, 보간, 순운동학/역운동학, 스키닝 |
| 13 | [파티클 시스템과 이펙트](13_Particle_Systems_and_Effects.md) | 이미터, 힘, 빌보드, GPU 파티클, 체적 효과 |
| 14 | [GPU 컴퓨팅](14_GPU_Computing.md) | 컴퓨트 셰이더, GPGPU, 병렬 패턴, 이미지 처리 |
| 15 | [실시간 렌더링 기법](15_Real_Time_Rendering_Techniques.md) | 지연 셰이딩, 그림자 맵, SSAO, 블룸, HDR, LOD |
| 16 | [현대 그래픽스 API 개요](16_Modern_Graphics_APIs_Overview.md) | Vulkan, Metal, DirectX 12, 명시적 자원 관리, 렌더 그래프 |

## 관련 토픽

| 토픽 | 연결 |
|------|------|
| Mathematical_Methods | 선형대수, 행렬 변환, 좌표계 |
| Optics | 빛 전달의 물리적 기초, 반사 모델, 렌즈 |
| Computer_Vision | 이미지 형성, 카메라 모델, 3D 재구성 |
| Deep_Learning | 신경 렌더링(NeRF), 미분가능 렌더링 |
| Web_Development | WebGL, Three.js, 캔버스 기반 그래픽스 |
| Signal_Processing | 안티앨리어싱, 텍스처 필터링, 이미지 처리 |

## 예제 파일

`examples/Computer_Graphics/`에 위치:

| 파일 | 설명 |
|------|------|
| `01_transformations_2d.py` | 2D 변환 행렬, 합성, 애니메이션 |
| `02_transformations_3d.py` | 3D 모델/뷰/투영, 카메라 궤도 |
| `03_rasterizer.py` | 소프트웨어 래스터라이저: 선 그리기, 삼각형 채우기, Z-버퍼 |
| `04_shading.py` | 퐁, 블린-퐁, PBR 셰이딩 비교 |
| `05_texture_mapping.py` | UV 매핑, 이중선형 필터링, 밉맵 |
| `06_webgl_triangle.html` | 최소 WebGL: 정점/프래그먼트 셰이더로 색상 삼각형 |
| `07_webgl_cube.html` | 조명과 텍스처가 있는 인터랙티브 3D 큐브 |
| `08_ray_tracer.py` | 간단한 재귀적 레이 트레이서: 구, 평면, 반사 |
| `09_path_tracer.py` | 중요도 샘플링 기반 몬테카를로 패스 트레이서 |
| `10_particle_system.py` | 힘과 빌보드를 가진 GPU 스타일 파티클 시스템 |
| `11_animation.py` | 키프레임 보간, 골격 애니메이션 데모 |
| `12_scene_graph.py` | BVH 탐색을 포함한 계층적 씬 그래프 |

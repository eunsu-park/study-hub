# Examples

A collection of executable example code corresponding to the study materials.

## Directory Structure

```
examples/
├── Algorithm/              # Python, C, C++ examples (89 files)
│   ├── python/             # Python implementation (29)
│   ├── c/                  # C implementation (29 + Makefile)
│   └── cpp/                # C++ implementation (29 + Makefile)
│
├── C_Programming/          # C project examples (88 files)
│   ├── 02_calculator/
│   ├── 03_number_guess/
│   ├── 04_address_book/    # Address book management
│   ├── 05_dynamic_array/
│   ├── 06_linked_list/
│   ├── 07_file_crypto/     # File encryption
│   ├── 08_stack_queue/     # Stack/Queue implementation
│   ├── 09_hash_table/      # Hash table
│   ├── 10_snake_game/      # Snake game
│   ├── 11_minishell/       # Mini shell
│   ├── 12_multithread/     # Multithreading
│   ├── 13_embedded_basic/
│   ├── 14_network/         # Network programming
│   ├── 15_ipc/             # IPC, Signals
│   └── 17_testing/         # Unit testing, Profiling
│
├── Compiler_Design/        # Compiler design Python examples (13 files)
│   ├── 01_lexer.py ~ 10_bytecode_vm.py  # Lexer to Bytecode VM
│   ├── 11_register_allocator.py         # Register allocation
│   └── 12_mini_compiler.py             # End-to-end mini compiler
│
├── Computer_Architecture/  # Architecture simulators (12 files)
│   ├── 02_number_systems.py ~ 07_cpu_datapath.py  # Number systems, logic circuits, CPU
│   ├── 10_assembly_sim.py ~ 12_branch_predictor.py # ISA, pipeline, branch prediction
│   └── 15_cache_sim.py, 16_tlb_sim.py             # Cache, TLB
│
├── CPP/                    # C++ advanced examples (37 files)
│   ├── 01_modern_cpp.cpp       # Modern C++ (C++17/20)
│   ├── 02_stl_containers.cpp   # STL containers & algorithms
│   ├── 03_smart_pointers.cpp   # Smart pointers
│   ├── 04_threading.cpp        # Multithreading
│   ├── 05_design_patterns.cpp  # Design patterns
│   ├── 06_templates.cpp        # Template metaprogramming
│   ├── 07_move_semantics.cpp   # Move semantics
│   ├── 08_cmake_demo/          # CMake demo project
│   ├── student_management/     # Student management project
│   └── Makefile                # Build system
│
├── Claude_Ecosystem/      # Claude ecosystem examples (16 files)
│   ├── 03_claude_md/          # CLAUDE.md, settings examples
│   ├── 05_hooks/              # Hook config examples
│   ├── 06_skills/             # Custom skills
│   ├── 07_subagents/          # Subagent definitions
│   ├── 13_mcp_server/         # MCP server implementations
│   ├── 16_tool_use/           # API tool use
│   └── 17_agent_sdk/          # Agent SDK examples
│
├── Control_Theory/        # Control theory Python examples (12 files)
│   ├── 01_modeling.py ~ 11_digital_control.py  # Modeling to Digital Control
│   └── README.md
│
├── Computer_Vision/        # OpenCV/Python examples (23 files)
├── Database_Theory/        # Database theory Python examples (15 files)
│   ├── 01-10_*.py                 # Relational model, B+Tree, MVCC, etc.
│   ├── 11_two_phase_locking.py    # 2PL concurrency
│   ├── 12_aries_recovery.py       # ARIES recovery
│   ├── 14_distributed_2pc.py      # Distributed 2PC
│   └── 16_design_case_study.py    # Design case study
│
├── Data_Science/           # Data science examples (27 files)
│   ├── data_analysis/      # NumPy, Pandas, visualization, Polars, DuckDB (10)
│   └── statistics/         # Statistics examples (17)
├── Data_Engineering/       # Airflow/Spark/Kafka/CDC/Lakehouse examples (33 files)
│   ├── airflow/            # TaskFlow API
│   ├── cdc/                # Debezium CDC
│   ├── kafka/              # Kafka Streams, ksqlDB
│   ├── lakehouse/          # Delta Lake patterns
│   ├── practical_pipeline/ # Practical pipeline project (L14)
│   └── spark/              # Structured Streaming
├── Deep_Learning/          # PyTorch examples (48 files)
│   ├── numpy/              # NumPy basic implementation (5)
│   ├── pytorch/            # PyTorch implementation (28)
│   └── implementations/   # Model implementation code (15)
│       ├── 01_Linear_Logistic/  # Linear/Logistic regression
│       ├── 03_CNN_LeNet/        # LeNet implementation
│       ├── 06_LSTM_GRU/         # LSTM/GRU implementation
│       └── ...                  # 12 model directories
│
├── Electrodynamics/       # Electrodynamics Python examples (12 files)
│
├── flagship/              # Flagship projects — self-contained single-file implementations (5 files)
│   ├── micro_autograd.py  # Autograd engine from scratch (numpy)
│   ├── tiny_gan.py        # GAN on 2D distributions (torch)
│   ├── nano_rl.py         # REINFORCE policy gradient gridworld (numpy)
│   ├── pico_diffusion.py  # Minimal DDPM diffusion model (torch)
│   └── micro_vae.py       # VAE with 2D latent space viz (torch)
│
├── Docker/                 # Docker/Kubernetes examples (15 files)
│   ├── 01_multi_stage/     # Multi-stage Docker build
│   ├── 02_compose/         # Docker Compose 3-tier stack
│   ├── 03_k8s/             # Kubernetes manifests
│   └── 04_ci_cd/           # GitHub Actions CI/CD pipeline
│
├── LaTeX/                  # LaTeX examples (19 files)
│   ├── 01_hello_world/        # First document
│   ├── 02_document_structure/ # Document structure
│   ├── 04_math_basics/        # Math basics
│   ├── 05_math_advanced/      # Advanced math
│   ├── 06_figures/            # Figures
│   ├── 07_tables/             # Tables
│   ├── 08_bibliography/       # Bibliography
│   ├── 10_tikz_basics/        # TikZ basics
│   ├── 11_tikz_advanced/      # Advanced TikZ
│   ├── 12_beamer/             # Beamer presentations
│   ├── 13_custom_commands/    # Custom commands
│   └── 16_projects/           # Projects
│
├── IoT_Embedded/           # Raspberry Pi/MQTT examples (17 files)
│   ├── edge_ai/            # TFLite, ONNX inference
│   ├── networking/         # WiFi, BLE, MQTT, HTTP
│   ├── projects/           # Smart home, Image analysis, Cloud IoT
│   └── raspberry_pi/       # GPIO, sensors
│
├── LLM_and_NLP/            # NLP/HuggingFace examples (15 files)
├── Machine_Learning/       # sklearn/Jupyter examples (25 files)
├── Math_for_AI/            # AI math Python examples (13 files)
├── MLOps/                  # MLflow/CI/CD/DVC/LLMOps examples (32 files)
│   ├── cicd/               # ML CI/CD pipeline
│   ├── dvc/                # DVC data version control
│   ├── feature_store/      # Feast feature store examples (L11)
│   ├── llmops/             # LLMOps monitoring
│   └── practical_project/  # E2E MLOps project (L12)
├── Numerical_Simulation/   # Numerical analysis Python examples (8 files)
├── Programming/           # Programming concepts examples (13 files)
│   ├── 02_paradigms/          # Paradigm comparison
│   ├── 05_oop/                # OOP, SOLID
│   ├── 06_functional/         # Functional programming
│   ├── 07_design_patterns/    # Design patterns
│   ├── 08_clean_code/         # Clean code refactoring
│   ├── 09_error_handling/     # Error handling
│   ├── 10_testing/            # Testing (pytest)
│   └── 12_concurrency/       # Concurrency (threading, asyncio)
│
├── Networking/             # Networking simulators (12 files)
│   ├── 02_osi_packet_builder.py ~ 09_routing_protocol_sim.py  # OSI, subnets, routing
│   ├── 10_tcp_state_machine.py ~ 13_http_client.py            # TCP, DNS, HTTP
│   └── 15_firewall_sim.py ~ 18_ipv6_demo.py                  # Firewall, IPv6
│
├── OS_Theory/              # OS theory simulators (12 files)
│   ├── 02_process_demo.py ~ 09_deadlock_detection.py  # Process, scheduling, synchronization
│   ├── 12_paging_sim.py, 15_page_replacement.py       # Memory management
│   └── 17_filesystem_sim.py, 18_ipc_demo.py           # Filesystem, IPC
│
├── PostgreSQL/             # SQL examples (18 files)
│   ├── 01-07_*.sql                     # SQL queries (CRUD, joins, window, FTS, RLS)
│   ├── 08_primary_standby_compose.yml  # Primary-Standby replication setup
│   ├── 09_primary_standby_setup.sh     # Automated replication setup script
│   ├── 10-18_*.sql                     # Functions, transactions, triggers, monitoring, JSON, optimization, windows, partitioning
│   └── README.md                       # PostgreSQL examples guide
├── Reinforcement_Learning/ # RL Python examples (14 files)
├── Formal_Languages/      # Formal languages examples (9 files)
│   ├── 01_dfa_simulator.py            # DFA simulator, minimization
│   ├── 02_nfa_subset_construction.py  # NFA, subset construction
│   ├── 03_regular_expressions.py      # Regex engine (Thompson)
│   ├── 04_pumping_lemma.py            # Pumping lemma
│   ├── 05_cfg_cyk_parser.py           # CFG, CNF, CYK parsing
│   ├── 06_pushdown_automaton.py       # PDA
│   ├── 07_turing_machine.py           # Turing machine
│   └── 08_chomsky_hierarchy.py        # Chomsky hierarchy
│
├── Foundation_Models/      # Foundation model examples (8 files)
├── Mathematical_Methods/  # Mathematical methods Python examples (13 files)
├── MHD/                   # MHD Python examples (32 files)
│   ├── 01_equilibria/         # Equilibria
│   ├── 02_stability/          # Stability analysis
│   ├── 03_instabilities/      # Instabilities
│   ├── 04_reconnection/       # Reconnection
│   ├── 05_turbulence/         # Turbulence
│   ├── 06_dynamo/             # Dynamo
│   ├── 07_astrophysics/       # Astrophysics
│   ├── 08_fusion/             # Fusion
│   ├── 09_solvers/            # Numerical solvers
│   └── 10_projects/           # Projects
│
├── Plasma_Physics/        # Plasma physics Python examples (26 files)
│   ├── 01_fundamentals/       # Fundamental parameters
│   ├── 02_particle_motion/    # Particle motion
│   ├── 03_kinetic/            # Kinetic theory
│   ├── 04_waves/              # Plasma waves
│   ├── 05_fluid/              # Fluid models
│   ├── 06_diagnostics/        # Diagnostics
│   └── 07_projects/           # Projects
│
├── Python/                # Advanced Python examples (16 files)
├── Security/              # Security Python examples (16 files)
│   ├── 02_cryptography/       # AES, RSA, ECDSA
│   ├── 03_hashing/            # SHA, bcrypt, HMAC
│   ├── 04_tls/                # TLS client, certificates
│   ├── 05_authentication/     # OAuth2, JWT, TOTP
│   ├── 06_authorization/      # RBAC middleware
│   ├── 07_owasp/              # Vulnerable + fixed code
│   ├── 08_injection/          # SQL injection, XSS defense
│   ├── 10_api_security/       # Rate limiter, CORS
│   ├── 11_secrets/            # Vault, .env management
│   ├── 13_testing/            # Bandit, security testing
│   ├── 15_secure_api/         # Flask secure API project
│   └── 16_scanner/            # Vulnerability scanner
│
├── Signal_Processing/      # Signal processing Python examples (19 files)
│   ├── 01_signals_classification.py  # Signal classification
│   ├── 02_convolution.py             # Convolution
│   ├── ...                           # 03-15: Fourier, sampling, FFT, Z-transform, filters, adaptive, image
│   └── README.md
│
├── Software_Engineering/   # Software engineering examples (11 files)
│   ├── 04_user_story_template.md   # User stories
│   ├── 05_uml_class_diagram.py     # UML class diagram
│   ├── 06_estimation_calculator.py # Estimation calculator
│   ├── 07_code_metrics.py          # Code metrics
│   ├── 10_gantt_chart.py           # Gantt chart + CPM
│   ├── 11_tech_debt_tracker.py     # Tech debt tracker
│   ├── 13_ci_cd_pipeline.yml       # CI/CD GitHub Actions
│   └── 14_adr_template.md          # ADR template
│
├── Shell_Script/           # Bash scripting examples (30 files)
│   ├── 02_parameter_expansion/  # Parameter expansion
│   ├── 03_arrays/               # Arrays
│   ├── 05_function_library/     # Function libraries
│   ├── 06_io_redirection/       # I/O redirection
│   ├── 08_regex/                # Regex
│   ├── 09_process_management/   # Process management
│   ├── 10_error_handling/       # Error handling
│   ├── 11_argument_parsing/     # Argument parsing
│   ├── 13_testing/              # Testing (Bats)
│   ├── 14_task_runner/          # Task runner
│   ├── 15_deployment/           # Deployment
│   └── 16_monitoring/           # Monitoring
│
├── System_Design/          # System design simulators (14 files)
│   ├── 04_load_balancer.py ~ 08_sharding_sim.py       # Load balancer, cache, hashing, sharding
│   ├── 10_eventual_consistency.py ~ 11_message_queue.py # Consistency, message queue
│   ├── 14_circuit_breaker.py ~ 16_raft_sim.py          # Circuit breaker, saga, Raft
│   └── 17_url_shortener.py ~ 20_inverted_index.py      # URL shortener, metrics, inverted index
│
└── Web_Development/        # HTML/CSS/JS projects (50 files)
    ├── 15_project_spa/         # Single Page Application demo
    │   ├── index.html          # Main HTML
    │   ├── style.css           # Responsive styles with animations
    │   ├── router.js           # Hash-based SPA router
    │   └── app.js              # Application logic and components
```

**Total example files: ~1,060** (55 topics)

## How to Build

### C/C++ Examples (Algorithm)

```bash
cd examples/Algorithm/c
make          # Build all
make clean    # Clean

cd examples/Algorithm/cpp
make          # Build all
```

### C Programming Examples

```bash
cd examples/C_Programming/<project>
make          # Build per project
make clean    # Clean
```

### C++ Examples

```bash
cd examples/CPP
make          # Build all
make modern   # Build modern C++
make run-01_modern_cpp  # Run example
make clean    # Clean
```

### Python Examples

```bash
python examples/Algorithm/python/01_complexity.py
python examples/Reinforcement_Learning/06_q_learning.py
```

### Jupyter Notebooks (Machine_Learning)

```bash
cd examples/Machine_Learning
jupyter notebook
```

## Examples by Topic

| Topic | Files | Language | Description |
|-------|-------|----------|-------------|
| Algorithm | 92 | Python, C, C++ | Data structures, Algorithms |
| C_Programming | 88 | C | System programming projects, Network, IPC, Testing |
| Claude_Ecosystem | 16 | Python/JSON | Claude Code, MCP Servers, Agent SDK |
| Compiler_Design | 13 | Python | Lexer, Parser, AST, Type Checker, Bytecode VM, Register Allocator, Mini Compiler |
| Computer_Architecture | 12 | Python | Number systems, IEEE754, Logic gates, ALU, CPU, Pipeline, Branch predictor, Cache, TLB |
| Control_Theory | 12 | Python | Transfer functions, Root locus, Bode/Nyquist, PID, State-space, LQR, Kalman, Digital control |
| Computer_Vision | 23 | Python | OpenCV, Image processing |
| CPP | 37 | C++ | Modern C++, STL, Smart Pointers, Threading, Design Patterns, CMake, Student Management Project |
| Data_Engineering | 33 | Python/SQL/YAML/JSON | Airflow, Spark, Kafka, CDC, Lakehouse, Practical Pipeline |
| Data_Science | 27 | Python | NumPy, Pandas, Visualization, Statistics, Bayesian, Causal inference, Survival analysis, GARCH, Polars, DuckDB |
| Database_Theory | 15 | Python | Relational model, Normalization, B+Tree, MVCC, Query Optimizer, 2PL, ARIES, 2PC |
| Deep_Learning | 47 | Python | PyTorch, CNN, RNN, Transformer, GAN, VAE, Diffusion, Model Implementations |
| Docker | 15 | Docker/YAML | Multi-stage build, Compose, Kubernetes, CI/CD |
| Formal_Languages | 9 | Python | DFA, NFA, Regex, Pumping Lemma, CFG/CYK, PDA, Turing Machine, Chomsky Hierarchy |
| Foundation_Models | 8 | Python | Scaling Laws, Tokenizer, LoRA, RAG, Quantization, Distillation |
| IoT_Embedded | 17 | Python | Raspberry Pi, MQTT, Edge AI |
| LaTeX | 19 | LaTeX | Document typesetting, Math, TikZ, Beamer, Bibliography, Build |
| Linux | 3 | Bash | Disaster recovery, Performance diagnostics |
| LLM_and_NLP | 15 | Python | BERT, GPT, RAG, LangChain |
| Machine_Learning | 25 | Python/Jupyter | sklearn, Classification, Regression, Ensemble, Feature Engineering, SHAP/LIME, AutoML, Anomaly Detection, Production ML, A/B Testing |
| Math_for_AI | 13 | Python | Linear Algebra, SVD/PCA, Optimization, Probability, Information Theory, Tensors, Graphs, Attention |
| Mathematical_Methods | 13 | Python | Series, Complex numbers, Linear Algebra, Fourier, ODE/PDE, Special Functions, Tensors |
| MHD | 32 | Python | Equilibria, Stability, Reconnection, Turbulence, Dynamo, Fusion |
| MLOps | 32 | Python/YAML/JSON | MLflow, CI/CD, DVC, LLMOps, Feast Feature Store, Practical Project |
| Numerical_Simulation | 8 | Python | Numerical analysis, ODE, Monte Carlo |
| Plasma_Physics | 26 | Python | Debye shielding, Particle motion, Landau damping, Waves, Fluid, Diagnostics |
| Programming | 13 | Python | Paradigms, OOP, Functional, Design Patterns, Clean Code, Testing, Concurrency |
| Networking | 12 | Python | OSI packets, Subnets, Routing, TCP state machine, DNS, HTTP, Firewall, IPv6 |
| OS_Theory | 12 | Python | Process, Threading, Scheduling, Synchronization, Deadlock, Paging, Page Replacement, Filesystem, IPC |
| PostgreSQL | 18 | SQL/Docker/Bash | CRUD, JOIN, Window functions, FTS, RLS, Functions, Transactions, Triggers, Monitoring, JSON, Optimization, Partitioning, Replication |
| Python | 16 | Python | Type Hints, Decorators, Generators, Async, Metaclasses, Testing |
| Reinforcement_Learning | 14 | Python | Q-Learning, DQN, PPO, A2C, Model-Based RL, SAC |
| Security | 16 | Python | Cryptography, Hashing, TLS, Auth, OWASP, Injection, API Security, CIA, Headers, Container, Incident |
| Shell_Script | 30 | Bash | Parameter expansion, Arrays, I/O, Regex, Process, Error handling, Testing, Deployment, Monitoring |
| Signal_Processing | 19 | Python | Signals, Convolution, Fourier, Sampling, FFT, Z-Transform, Filter Design, Multirate, Adaptive, Spectrogram, Image Filtering |
| Software_Engineering | 11 | Python/MD/YAML | User Stories, UML, Estimation, Code Metrics, Gantt Chart, Tech Debt, CI/CD, ADR, Testing, Branching |
| System_Design | 14 | Python | Load Balancer, Rate Limiter, Cache, Hashing, Sharding, Consistency, Message Queue, Circuit Breaker, Raft, URL Shortener, Inverted Index |
| VIM | 10 | Vim script/Config | Vim modes, motions, macros, plugins, vimrc |
| Web_Development | 50 | HTML/CSS/JS/TS | Web projects, TypeScript, SPA router |
| **Flagship** | **5** | **Python** | **Self-contained single-file projects: autograd, GAN, RL, diffusion, VAE** |
| --- | --- | --- | --- |
| Backend_Frameworks | 12 | Python/JS | FastAPI, Express, Django examples |
| Calculus_and_Differential_Equations | 14 | Python | Limits, derivatives, integrals, ODE/PDE |
| Cloud_Computing | 10 | Python/YAML | AWS, GCP, Terraform, multi-cloud |
| Computer_Graphics | 12 | Python | Ray tracing, rasterization, shaders, 3D transforms |
| Cryptography_Theory | 13 | Python | RSA, ECC, lattice, zero-knowledge proofs |
| Electrodynamics | 12 | Python | Maxwell's equations, EM waves, waveguides |
| Frontend_Frameworks | 7 | JS/TS | React, Vue, Svelte examples |
| Git | 10 | Bash/Config | Branching, merging, hooks, workflows |
| GraphQL | 7 | Python/JS | Schema, resolvers, subscriptions |
| Optics | 13 | Python | Geometric optics, diffraction, interferometry, Zernike |
| Quantum_Computing | 13 | Python | Qubits, gates, entanglement, Shor, VQE |
| Robotics | 12 | Python | Kinematics, path planning, SLAM, control |
| Rust | 11 | Rust | Ownership, traits, concurrency, async, macros |
| Solar_Physics | 12 | Python | Solar structure, corona, flares, CME |
| Space_Weather | 12 | Python | Magnetosphere, auroras, space weather forecasting |

## Mapping Examples to Study Materials

Example files correspond to study materials in `content/`.

Examples:
- `content/en/Algorithm/01_Complexity_Analysis.md` -> `examples/Algorithm/python/01_complexity.py`
- `content/en/C_Programming/05_Project_Address_Book.md` -> `examples/C_Programming/04_address_book/`
- `content/en/Machine_Learning/04_Model_Evaluation.md` -> `examples/Machine_Learning/03_model_evaluation.ipynb`

# Examples

A collection of executable example code corresponding to the study materials.

## Directory Structure

```
examples/
├── Algorithm/              # Python, C, C++ examples (95 files)
│   ├── python/             # Python implementation (29)
│   ├── c/                  # C implementation (29 + Makefile)
│   └── cpp/                # C++ implementation (29 + Makefile)
│
├── C_Programming/          # C project examples (75 files)
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
├── Claude_Ecosystem/      # Claude ecosystem examples (25 files)
│   ├── 03_claude_md/          # CLAUDE.md, settings examples
│   ├── 04_permissions/        # Permission rules
│   ├── 05_hooks/              # Hook config examples
│   ├── 06_skills/             # Custom skills
│   ├── 07_subagents/          # Subagent definitions
│   ├── 08_agent_teams/        # Agent team orchestration
│   ├── 12_mcp_basics/         # MCP client config
│   ├── 13_mcp_server/         # MCP server implementations
│   ├── 15_api/                # API messages
│   ├── 16_tool_use/           # API tool use
│   ├── 17_agent_sdk/          # Agent SDK examples
│   ├── 18_custom_agents/      # Custom agents with guardrails
│   ├── 19_optimization/       # Cost calculator
│   ├── 20_workflows/          # Development workflows
│   └── 21_best_practices/     # Prompt patterns
│
├── Compiler_Design/        # Compiler design Python examples (12 files)
│   ├── 01_lexer.py ~ 10_bytecode_vm.py  # Lexer to Bytecode VM
│   ├── 11_register_allocator.py         # Register allocation
│   └── 12_mini_compiler.py             # End-to-end mini compiler
│
├── Computer_Architecture/  # Architecture simulators (13 files)
│   ├── 02_number_systems.py ~ 07_cpu_datapath.py  # Number systems, logic circuits, CPU
│   ├── 10_assembly_sim.py ~ 12_branch_predictor.py # ISA, pipeline, branch prediction
│   └── 15_cache_sim.py, 16_tlb_sim.py             # Cache, TLB
│
├── CPP/                    # C++ advanced examples (34 files)
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
├── Control_Theory/        # Control theory Python examples (11 files)
│   ├── 01_modeling.py ~ 11_digital_control.py  # Modeling to Digital Control
│   └── README.md
│
├── Computer_Vision/        # OpenCV/Python examples (22 files)
├── Data_Science/           # Data science examples (27 files)
│   ├── data_analysis/      # NumPy, Pandas, visualization, Polars, DuckDB (10)
│   └── statistics/         # Statistics examples (17)
├── Data_Engineering/       # Airflow/Spark/Kafka/CDC/Lakehouse examples (36 files)
│   ├── airflow/            # TaskFlow API
│   ├── cdc/                # Debezium CDC
│   ├── kafka/              # Kafka Streams, ksqlDB
│   ├── lakehouse/          # Delta Lake patterns
│   ├── practical_pipeline/ # Practical pipeline project (L14)
│   └── spark/              # Structured Streaming
├── Database_Theory/        # Database theory Python examples (14 files)
│   ├── 01-10_*.py                 # Relational model, B+Tree, MVCC, etc.
│   ├── 11_two_phase_locking.py    # 2PL concurrency
│   ├── 12_aries_recovery.py       # ARIES recovery
│   ├── 14_distributed_2pc.py      # Distributed 2PC
│   └── 16_design_case_study.py    # Design case study
│
├── Deep_Learning/          # PyTorch examples (61 files)
│   ├── numpy/              # NumPy basic implementation (5)
│   ├── pytorch/            # PyTorch implementation (28)
│   └── implementations/   # Model implementation code (28)
│       ├── 01_Linear_Logistic/  # Linear/Logistic regression
│       ├── 03_CNN_LeNet/        # LeNet implementation
│       ├── 06_LSTM_GRU/         # LSTM/GRU implementation
│       └── ...                  # 12 model directories
│
├── Docker/                 # Docker/Kubernetes examples (14 files)
│   ├── 01_multi_stage/     # Multi-stage Docker build
│   ├── 02_compose/         # Docker Compose 3-tier stack
│   ├── 03_k8s/             # Kubernetes manifests
│   └── 04_ci_cd/           # GitHub Actions CI/CD pipeline
│
├── Electrodynamics/       # Electrodynamics Python examples (12 files)
│
├── Flagship/              # Flagship projects — self-contained single-file implementations (10 files)
│   ├── micro_autograd.py  # Autograd engine from scratch (numpy)
│   ├── tiny_gan.py        # GAN on 2D distributions (torch)
│   ├── nano_rl.py         # REINFORCE policy gradient gridworld (numpy)
│   ├── pico_diffusion.py  # Minimal DDPM diffusion model (torch)
│   └── micro_vae.py       # VAE with 2D latent space viz (torch)
│
├── GraphQL/               # GraphQL examples (12 files)
│   ├── 01_schema_resolvers.js ~ 07_testing.js  # Schema, DataLoader, Auth, Subscriptions, Testing
│   ├── 08_persisted_queries.js ~ 10_performance_security.js  # Caching, Federation, Security
│   └── 11_rest_migration.js, 12_api_gateway.js              # Migration, Gateway
│
├── IoT_Embedded/           # Raspberry Pi/MQTT examples (16 files)
│   ├── edge_ai/            # TFLite, ONNX inference
│   ├── networking/         # WiFi, BLE, MQTT, HTTP
│   ├── projects/           # Smart home, Image analysis, Cloud IoT
│   └── raspberry_pi/       # GPIO, sensors
│
├── LaTeX/                  # LaTeX examples (17 files)
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
├── LLM_and_NLP/            # NLP/HuggingFace examples (15 files)
├── Linux/                  # Linux administration examples (8 files)
├── Machine_Learning/       # sklearn/Jupyter examples (26 files)
├── Math_for_AI/            # AI math Python examples (12 files)
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
├── MLOps/                  # MLflow/CI/CD/DVC/LLMOps examples (34 files)
│   ├── cicd/               # ML CI/CD pipeline
│   ├── dvc/                # DVC data version control
│   ├── feature_store/      # Feast feature store examples (L11)
│   ├── llmops/             # LLMOps monitoring
│   └── practical_project/  # E2E MLOps project (L12)
├── Networking/             # Networking simulators (14 files)
│   ├── 02_osi_packet_builder.py ~ 09_routing_protocol_sim.py  # OSI, subnets, routing
│   ├── 10_tcp_state_machine.py ~ 13_http_client.py            # TCP, DNS, HTTP
│   └── 15_firewall_sim.py ~ 18_ipv6_demo.py                  # Firewall, IPv6
│
├── Numerical_Simulation/   # Numerical analysis Python examples (14 files)
├── OS_Theory/              # OS theory simulators (11 files)
│   ├── 02_process_demo.py ~ 09_deadlock_detection.py  # Process, scheduling, synchronization
│   ├── 12_paging_sim.py, 15_page_replacement.py       # Memory management
│   └── 17_filesystem_sim.py, 18_ipc_demo.py           # Filesystem, IPC
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
├── PostgreSQL/             # SQL examples (17 files)
│   ├── 01-07_*.sql                     # SQL queries (CRUD, joins, window, FTS, RLS)
│   ├── 08_primary_standby_compose.yml  # Primary-Standby replication setup
│   ├── 09_primary_standby_setup.sh     # Automated replication setup script
│   └── 10-18_*.sql                     # Functions, transactions, triggers, monitoring, JSON
│
├── Probability_and_Statistics/  # Probability/statistics Python examples (18 files)
│   ├── 01_combinatorics.py ~ 05_joint_distributions.py  # Counting, axioms, random variables, moments, joint
│   ├── 06_discrete_distributions.py ~ 09_multivariate_normal.py  # Distribution families, transformations, MVN
│   ├── 10_convergence.py ~ 12_point_estimation.py  # Convergence, LLN/CLT, estimation
│   └── 13_interval_estimation.py ~ 18_stochastic_processes.py  # CI, testing, Bayesian, nonparametric, regression, stochastic
│
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
├── Python/                # Advanced Python examples (16 files)
├── Reinforcement_Learning/ # RL Python examples (16 files)
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
├── Shell_Script/           # Bash scripting examples (29 files)
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
├── Signal_Processing/      # Signal processing Python examples (18 files)
│   ├── 01_signals_classification.py  # Signal classification
│   ├── 02_convolution.py             # Convolution
│   └── ...                           # 03-18: Fourier, sampling, FFT, Z-transform, filters, adaptive, image
│
├── Software_Engineering/   # Software engineering examples (16 files)
│   ├── 01_se_principles.py       # Core SE principles
│   ├── 02_sdlc_models.py         # SDLC models
│   ├── 03_agile_simulator.py     # Agile development simulator
│   ├── 04_user_story_template.md # User stories
│   ├── 05_uml_class_diagram.py   # UML class diagram
│   ├── 06_estimation_calculator.py # Estimation calculator
│   ├── 07_code_metrics.py        # Code metrics
│   ├── 08_test_plan_template.md  # Test plan template
│   ├── 09_branching_strategy.py  # Git branching strategies
│   ├── 10_gantt_chart.py         # Gantt chart + CPM
│   ├── 11_tech_debt_tracker.py   # Tech debt tracker
│   ├── 12_process_improvement.py # Process improvement
│   ├── 13_ci_cd_pipeline.yml     # CI/CD GitHub Actions
│   ├── 14_adr_template.md        # ADR template
│   ├── 15_team_dynamics.py       # Team dynamics
│   └── 16_capstone_project.md    # Capstone project
│
├── Solar_Physics/         # Solar physics Python examples (24 files)
├── Space_Weather/         # Space weather Python examples (24 files)
├── System_Design/          # System design simulators (13 files)
│   ├── 04_load_balancer.py ~ 08_sharding_sim.py       # Load balancer, cache, hashing, sharding
│   ├── 10_eventual_consistency.py ~ 11_message_queue.py # Consistency, message queue
│   ├── 14_circuit_breaker.py ~ 16_raft_sim.py          # Circuit breaker, saga, Raft
│   └── 17_url_shortener.py ~ 20_inverted_index.py      # URL shortener, metrics, inverted index
│
├── VIM/                    # Vim practice files and configs (16 files)
│   ├── 01_basic_motions.txt ~ 05_macro_examples.txt   # Practice files
│   ├── 06_minimal_vimrc.vim ~ 08_advanced_vimrc.vim   # Vimscript configs
│   ├── 09_init_lua.lua         # Neovim Lua config
│   ├── 10_vim_cheatsheet.md    # Command cheatsheet
│   ├── 11_modes_practice.txt ~ 15_command_line_advanced.txt  # More practice
│   └── 16_plugins_guide.md    # Plugins and ecosystem
│
└── Web_Development/        # HTML/CSS/JS projects (49 files)
    ├── 15_project_spa/         # Single Page Application demo
    │   ├── index.html          # Main HTML
    │   ├── style.css           # Responsive styles with animations
    │   ├── router.js           # Hash-based SPA router
    │   └── app.js              # Application logic and components
```

**Total example files: ~1,157** (55 topics + Flagship)

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
python examples/Probability_and_Statistics/01_combinatorics.py
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
| Algorithm | 95 | Python, C, C++ | Data structures, Algorithms |
| Backend_Frameworks | 12 | Python/JS | FastAPI, Express, Django examples |
| C_Programming | 75 | C | System programming projects, Network, IPC, Testing |
| Calculus_and_Differential_Equations | 14 | Python | Limits, derivatives, integrals, ODE/PDE |
| Claude_Ecosystem | 25 | Python/JSON/MD | Claude Code, MCP Servers, Agent SDK, Hooks, Permissions |
| Cloud_Computing | 10 | Python/YAML | AWS, GCP, Terraform, multi-cloud |
| Compiler_Design | 12 | Python | Lexer, Parser, AST, Type Checker, Bytecode VM, Register Allocator, Mini Compiler |
| Computer_Architecture | 13 | Python | Number systems, IEEE754, Logic gates, ALU, CPU, Pipeline, Branch predictor, Cache, TLB |
| Computer_Graphics | 12 | Python | Ray tracing, rasterization, shaders, 3D transforms |
| Computer_Vision | 22 | Python | OpenCV, Image processing, Object detection, 3D vision |
| Control_Theory | 11 | Python | Transfer functions, Root locus, Bode/Nyquist, PID, State-space, LQR, Kalman, Digital control |
| CPP | 34 | C++ | Modern C++, STL, Smart Pointers, Threading, Design Patterns, CMake, Student Management Project |
| Cryptography_Theory | 13 | Python | RSA, ECC, lattice, zero-knowledge proofs |
| Data_Engineering | 36 | Python/SQL/YAML/JSON | Airflow, Spark, Kafka, CDC, Lakehouse, Practical Pipeline |
| Data_Science | 27 | Python | NumPy, Pandas, Visualization, Statistics, Bayesian, Causal inference, Polars, DuckDB |
| Database_Theory | 14 | Python | Relational model, Normalization, B+Tree, MVCC, Query Optimizer, 2PL, ARIES, 2PC |
| Deep_Learning | 61 | Python | PyTorch, CNN, RNN, Transformer, GAN, VAE, Diffusion, Model Implementations |
| Docker | 14 | Docker/YAML | Multi-stage build, Compose, Kubernetes, CI/CD |
| Electrodynamics | 12 | Python | Maxwell's equations, EM waves, waveguides |
| Formal_Languages | 8 | Python | DFA, NFA, Regex, Pumping Lemma, CFG/CYK, PDA, Turing Machine, Chomsky Hierarchy |
| Foundation_Models | 12 | Python | Scaling Laws, Tokenizer, LoRA, RAG, Quantization, Distillation |
| Frontend_Frameworks | 7 | JS/TS | React, Vue, Svelte examples |
| Git | 10 | Bash/Config | Branching, merging, hooks, workflows |
| GraphQL | 12 | JS/Python/TSX | Schema, DataLoader, Auth, Subscriptions, Federation, Testing, API Gateway |
| IoT_Embedded | 16 | Python | Raspberry Pi, MQTT, Edge AI |
| LaTeX | 17 | LaTeX | Document typesetting, Math, TikZ, Beamer, Bibliography, Build |
| Linux | 8 | Bash | System administration, Disaster recovery, Performance diagnostics |
| LLM_and_NLP | 15 | Python | BERT, GPT, RAG, LangChain |
| Machine_Learning | 26 | Python/Jupyter | sklearn, Classification, Regression, Ensemble, SHAP/LIME, AutoML, Symbolic Regression |
| Math_for_AI | 12 | Python | Linear Algebra, SVD/PCA, Optimization, Probability, Information Theory, Tensors, Attention |
| Mathematical_Methods | 12 | Python | Series, Complex numbers, Linear Algebra, Fourier, ODE/PDE, Special Functions, Tensors |
| MHD | 32 | Python | Equilibria, Stability, Reconnection, Turbulence, Dynamo, Fusion |
| MLOps | 34 | Python/YAML/JSON | MLflow, CI/CD, DVC, LLMOps, Feast Feature Store, Practical Project |
| Networking | 14 | Python | OSI packets, Subnets, Routing, TCP state machine, DNS, HTTP, Firewall, IPv6 |
| Numerical_Simulation | 14 | Python | Numerical analysis, ODE/PDE, CFD, FDTD, Monte Carlo |
| Optics | 13 | Python | Geometric optics, diffraction, interferometry, Zernike |
| OS_Theory | 11 | Python | Process, Threading, Scheduling, Synchronization, Deadlock, Paging, Filesystem, IPC |
| Plasma_Physics | 26 | Python | Debye shielding, Particle motion, Landau damping, Waves, Fluid, Diagnostics |
| PostgreSQL | 17 | SQL/Docker/Bash | CRUD, JOIN, Window functions, FTS, RLS, Transactions, Replication, Partitioning |
| Probability_and_Statistics | 18 | Python | Combinatorics, Distributions, CLT, Estimation, Hypothesis Testing, Bayesian, Stochastic Processes |
| Programming | 13 | Python | Paradigms, OOP, Functional, Design Patterns, Clean Code, Testing, Concurrency |
| Python | 16 | Python | Type Hints, Decorators, Generators, Async, Metaclasses, Testing |
| Quantum_Computing | 13 | Python | Qubits, gates, entanglement, Shor, VQE |
| Reinforcement_Learning | 16 | Python | Q-Learning, DQN, PPO, A2C, Model-Based RL, SAC |
| Robotics | 12 | Python | Kinematics, path planning, SLAM, control |
| Rust | 11 | Rust | Ownership, traits, concurrency, async, macros |
| Security | 16 | Python | Cryptography, Hashing, TLS, Auth, OWASP, Injection, API Security |
| Shell_Script | 29 | Bash | Parameter expansion, Arrays, I/O, Regex, Process, Error handling, Deployment, Monitoring |
| Signal_Processing | 18 | Python | Signals, Convolution, Fourier, FFT, Z-Transform, Filter Design, Adaptive, Image Filtering |
| Software_Engineering | 16 | Python/MD/YAML | SE Principles, SDLC, Agile, UML, Estimation, Metrics, CI/CD, Team Dynamics |
| Solar_Physics | 24 | Python | Solar structure, corona, flares, CME |
| Space_Weather | 24 | Python | Magnetosphere, auroras, space weather forecasting |
| System_Design | 13 | Python | Load Balancer, Cache, Sharding, Consistency, Message Queue, Circuit Breaker, Raft |
| VIM | 16 | Vim script/Config | Vim modes, motions, macros, plugins, vimrc, Neovim |
| Web_Development | 49 | HTML/CSS/JS/TS | Web projects, TypeScript, SPA router |
| **Flagship** | **10** | **Python** | **Self-contained single-file projects: autograd, GAN, RL, diffusion, VAE** |

## Mapping Examples to Study Materials

Example files correspond to study materials in `content/`.

Examples:
- `content/en/Algorithm/01_Complexity_Analysis.md` -> `examples/Algorithm/python/01_complexity.py`
- `content/en/C_Programming/05_Project_Address_Book.md` -> `examples/C_Programming/04_address_book/`
- `content/en/Machine_Learning/04_Model_Evaluation.md` -> `examples/Machine_Learning/03_model_evaluation.ipynb`
- `content/en/Probability_and_Statistics/05_Joint_Distributions.md` -> `examples/Probability_and_Statistics/05_joint_distributions.py`

"""
Modern Graphics APIs -- Command Buffer and Render Graph Simulation
===================================================================

Simulates the explicit resource management model of modern graphics APIs
(Vulkan, Metal, DirectX 12, WebGPU):
  1. GPU resource management (buffers, textures, render targets)
  2. Command buffer recording and submission
  3. Pipeline state objects (PSO)
  4. Synchronization primitives (fences, barriers)
  5. Render graph -- automatic resource lifetime and barrier insertion
  6. Multi-pass rendering orchestration

Modern APIs give applications explicit control over GPU resources and
synchronization.  This simulation shows *why* that matters: the render
graph can optimize resource usage and insert barriers automatically,
something the old OpenGL driver did (poorly) behind the scenes.

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Callable, Tuple

# ---------------------------------------------------------------------------
# 1. Resource types -- what the GPU manages
# ---------------------------------------------------------------------------


class ResourceState(Enum):
    """Resource states that determine how the GPU can access a resource.

    Why explicit states?  In Vulkan/DX12, the application must declare
    how a resource will be used BEFORE using it.  The driver inserts
    memory barriers at state transitions to ensure cache coherency.
    Getting this wrong causes rendering corruption.
    """
    UNDEFINED = auto()
    RENDER_TARGET = auto()    # Being written to as a color attachment
    DEPTH_STENCIL = auto()    # Being written to as depth/stencil
    SHADER_READ = auto()      # Being sampled in a shader
    TRANSFER_SRC = auto()     # Source of a copy operation
    TRANSFER_DST = auto()     # Destination of a copy operation
    PRESENT = auto()          # Ready to display on screen


@dataclass
class GPUResource:
    """A GPU resource (buffer or texture) with tracked state.

    Why track state?  Modern APIs require the application to manage
    resource lifetimes and transitions.  The old APIs (OpenGL) did this
    inside the driver, but the driver couldn't predict future usage and
    often inserted unnecessary barriers, hurting performance.
    """
    name: str
    width: int = 0
    height: int = 0
    channels: int = 4
    state: ResourceState = ResourceState.UNDEFINED
    data: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.data is None and self.width > 0 and self.height > 0:
            self.data = np.zeros((self.height, self.width, self.channels),
                                 dtype=float)


# ---------------------------------------------------------------------------
# 2. Pipeline State Object (PSO)
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """A Pipeline State Object encapsulates all GPU state for a draw call.

    Why bundle everything into one object?  In OpenGL, state is set
    individually (glEnable, glBlendFunc, glUseProgram...) and the driver
    must validate the combination at draw time.  PSOs are pre-validated
    at creation time -- draw calls just bind a PSO and go.

    This eliminates thousands of validation checks per frame and allows
    the driver to pre-compile the hardware state.
    """
    name: str
    vertex_shader: str = "default_vs"
    fragment_shader: str = "default_fs"
    depth_test: bool = True
    depth_write: bool = True
    blend_enabled: bool = False
    cull_mode: str = "back"
    topology: str = "triangles"

    def __repr__(self):
        return (f"PSO({self.name}: vs={self.vertex_shader}, "
                f"fs={self.fragment_shader}, depth={'on' if self.depth_test else 'off'})")


# ---------------------------------------------------------------------------
# 3. Command Buffer -- deferred GPU command recording
# ---------------------------------------------------------------------------


class CommandType(Enum):
    BARRIER = auto()
    BIND_PIPELINE = auto()
    BIND_RESOURCE = auto()
    SET_RENDER_TARGET = auto()
    DRAW = auto()
    DISPATCH_COMPUTE = auto()
    COPY = auto()
    CLEAR = auto()


@dataclass
class GPUCommand:
    """A single recorded GPU command."""
    cmd_type: CommandType
    description: str
    resource: Optional[str] = None
    details: Optional[dict] = None


class CommandBuffer:
    """Records GPU commands for later submission.

    Why record now, submit later?  Modern APIs separate recording from
    execution.  This enables:
    1. Multi-threaded recording (each thread records its own buffer)
    2. Command reuse (record once, submit every frame)
    3. Driver optimization (the driver sees all commands before executing)

    In Vulkan, you create a VkCommandBuffer, call vkBeginCommandBuffer,
    record commands, call vkEndCommandBuffer, then submit to a queue.
    """

    def __init__(self, name: str = "primary"):
        self.name = name
        self.commands: List[GPUCommand] = []
        self.is_recording = False

    def begin(self):
        """Begin recording commands."""
        self.commands = []
        self.is_recording = True

    def end(self):
        """Finish recording."""
        self.is_recording = False

    def barrier(self, resource_name: str,
                old_state: ResourceState, new_state: ResourceState):
        """Insert a pipeline barrier (state transition).

        Barriers are the most critical part of modern API programming.
        They tell the GPU: "finish all writes to this resource, flush
        caches, then allow reads in the new layout."  Missing a barrier
        causes data races on the GPU.
        """
        self.commands.append(GPUCommand(
            cmd_type=CommandType.BARRIER,
            description=f"Barrier: {resource_name} {old_state.name} -> {new_state.name}",
            resource=resource_name,
            details={'old': old_state, 'new': new_state}
        ))

    def bind_pipeline(self, pso: PipelineState):
        self.commands.append(GPUCommand(
            cmd_type=CommandType.BIND_PIPELINE,
            description=f"Bind PSO: {pso.name}"
        ))

    def set_render_target(self, color_target: str,
                          depth_target: Optional[str] = None):
        self.commands.append(GPUCommand(
            cmd_type=CommandType.SET_RENDER_TARGET,
            description=f"Set RT: color={color_target}, depth={depth_target}",
            resource=color_target,
            details={'depth': depth_target}
        ))

    def clear(self, target: str, color: Tuple[float, ...] = (0, 0, 0, 1)):
        self.commands.append(GPUCommand(
            cmd_type=CommandType.CLEAR,
            description=f"Clear: {target}",
            resource=target,
            details={'color': color}
        ))

    def draw(self, vertex_count: int, description: str = ""):
        self.commands.append(GPUCommand(
            cmd_type=CommandType.DRAW,
            description=f"Draw: {vertex_count} verts ({description})"
        ))

    def dispatch_compute(self, groups_x: int, groups_y: int, groups_z: int,
                         description: str = ""):
        self.commands.append(GPUCommand(
            cmd_type=CommandType.DISPATCH_COMPUTE,
            description=(f"Dispatch: ({groups_x}, {groups_y}, {groups_z}) "
                         f"({description})")
        ))

    def copy(self, src: str, dst: str):
        self.commands.append(GPUCommand(
            cmd_type=CommandType.COPY,
            description=f"Copy: {src} -> {dst}",
            resource=dst,
            details={'src': src}
        ))

    def print_commands(self):
        """Display all recorded commands (useful for debugging)."""
        print(f"\n  Command Buffer '{self.name}' ({len(self.commands)} commands):")
        for i, cmd in enumerate(self.commands):
            prefix = "    " if cmd.cmd_type != CommandType.BARRIER else "  > "
            print(f"  {prefix}[{i:2d}] {cmd.description}")


# ---------------------------------------------------------------------------
# 4. Synchronization Primitives
# ---------------------------------------------------------------------------


@dataclass
class Fence:
    """CPU-GPU synchronization: CPU waits until GPU finishes.

    Why fences?  The CPU submits work to the GPU and continues.  When
    the CPU needs a result (e.g., to read back a screenshot), it waits
    on a fence.  Without fences, the CPU might read incomplete data.
    """
    name: str
    signaled: bool = False

    def signal(self):
        self.signaled = True

    def wait(self):
        """Block until the fence is signaled (simulated)."""
        if not self.signaled:
            print(f"    Fence '{self.name}': waiting for GPU...")
        self.signaled = False

    def is_ready(self) -> bool:
        return self.signaled


@dataclass
class Semaphore:
    """GPU-GPU synchronization between queue submissions.

    Semaphores coordinate work between different GPU queues (e.g.,
    the graphics queue must wait for the compute queue to finish).
    Unlike fences, semaphores are GPU-only -- the CPU never blocks.
    """
    name: str
    signaled: bool = False


# ---------------------------------------------------------------------------
# 5. Render Graph -- automatic resource management
# ---------------------------------------------------------------------------


@dataclass
class RenderPass:
    """A render pass in the render graph.

    The render graph is a directed acyclic graph where each node is a
    render pass.  Edges represent resource dependencies.  The graph
    compiler determines:
    1. Execution order (topological sort)
    2. Resource lifetimes (allocate late, free early)
    3. Barrier placement (only where actually needed)
    """
    name: str
    reads: List[str] = field(default_factory=list)
    writes: List[str] = field(default_factory=list)
    execute_fn: Optional[Callable] = None


class RenderGraph:
    """A frame-level render graph that manages passes and resources.

    Why a render graph?  Manually tracking barriers and resource states
    across dozens of render passes is error-prone.  The render graph
    automates this: you declare what each pass reads and writes, and the
    graph compiler handles the rest.

    Used by: Frostbite (EA), Unreal Engine 5, Unity HDRP.
    """

    def __init__(self):
        self.passes: List[RenderPass] = []
        self.resources: Dict[str, GPUResource] = {}

    def add_resource(self, resource: GPUResource):
        self.resources[resource.name] = resource

    def add_pass(self, render_pass: RenderPass):
        self.passes.append(render_pass)

    def compile(self) -> List[RenderPass]:
        """Topological sort of passes based on read/write dependencies.

        A pass that reads resource X must execute after any pass that
        writes resource X.  This is the same constraint-solving problem
        as instruction scheduling in a compiler.
        """
        # Build dependency graph
        writer_of: Dict[str, int] = {}
        for i, p in enumerate(self.passes):
            for w in p.writes:
                writer_of[w] = i

        adj: Dict[int, List[int]] = {i: [] for i in range(len(self.passes))}
        in_degree = [0] * len(self.passes)

        for i, p in enumerate(self.passes):
            for r in p.reads:
                if r in writer_of:
                    dep = writer_of[r]
                    if dep != i:
                        adj[dep].append(i)
                        in_degree[i] += 1

        # Kahn's algorithm (topological sort)
        queue = [i for i in range(len(self.passes)) if in_degree[i] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.passes):
            raise RuntimeError("Render graph has a cycle (impossible to execute)")

        return [self.passes[i] for i in order]

    def compute_barriers(self, ordered_passes: List[RenderPass]) -> List[dict]:
        """Determine which barriers are needed between passes.

        For each resource, track its current state.  When a pass needs
        the resource in a different state, insert a barrier.
        """
        barriers = []
        resource_states: Dict[str, ResourceState] = {
            name: ResourceState.UNDEFINED for name in self.resources
        }

        for p in ordered_passes:
            pass_barriers = []
            for r in p.reads:
                if resource_states.get(r) != ResourceState.SHADER_READ:
                    pass_barriers.append({
                        'resource': r,
                        'from': resource_states.get(r, ResourceState.UNDEFINED),
                        'to': ResourceState.SHADER_READ
                    })
                    resource_states[r] = ResourceState.SHADER_READ

            for w in p.writes:
                if resource_states.get(w) != ResourceState.RENDER_TARGET:
                    pass_barriers.append({
                        'resource': w,
                        'from': resource_states.get(w, ResourceState.UNDEFINED),
                        'to': ResourceState.RENDER_TARGET
                    })
                    resource_states[w] = ResourceState.RENDER_TARGET

            barriers.append({'pass': p.name, 'barriers': pass_barriers})

        return barriers

    def compute_lifetimes(self, ordered_passes: List[RenderPass]) -> Dict[str, Tuple[int, int]]:
        """Compute resource lifetimes (first use to last use).

        Why compute lifetimes?  Resources that don't overlap in time
        can share the same GPU memory (aliasing).  This is critical for
        reducing memory usage in complex frames with many render targets.
        """
        first_use: Dict[str, int] = {}
        last_use: Dict[str, int] = {}

        for i, p in enumerate(ordered_passes):
            for r in p.reads + p.writes:
                if r not in first_use:
                    first_use[r] = i
                last_use[r] = i

        return {r: (first_use[r], last_use[r]) for r in first_use}


# ---------------------------------------------------------------------------
# 6. Demonstration: Full frame with render graph
# ---------------------------------------------------------------------------


def build_demo_frame() -> RenderGraph:
    """Build a render graph for a typical deferred rendering frame.

    This represents the render passes in a modern game engine frame:
      1. Shadow map pass (depth-only from light's view)
      2. G-buffer pass (write albedo, normal, depth)
      3. SSAO pass (read depth/normal, write AO texture)
      4. Lighting pass (read G-buffer + shadow + AO, write HDR color)
      5. Bloom pass (read HDR, write bloom texture)
      6. Tone mapping (read HDR + bloom, write final LDR)
      7. UI overlay (read LDR, write to swapchain)
    """
    rg = RenderGraph()

    # Create resources
    resources = [
        GPUResource("shadow_map", 1024, 1024, 1),
        GPUResource("gbuf_albedo", 1920, 1080, 4),
        GPUResource("gbuf_normal", 1920, 1080, 4),
        GPUResource("gbuf_depth", 1920, 1080, 1),
        GPUResource("ssao_texture", 960, 540, 1),   # Half-res
        GPUResource("hdr_color", 1920, 1080, 4),
        GPUResource("bloom_texture", 480, 270, 4),   # Quarter-res
        GPUResource("ldr_output", 1920, 1080, 4),
        GPUResource("swapchain", 1920, 1080, 4),
    ]
    for r in resources:
        rg.add_resource(r)

    # Define passes
    rg.add_pass(RenderPass(
        name="Shadow",
        reads=[],
        writes=["shadow_map"]
    ))
    rg.add_pass(RenderPass(
        name="G-Buffer",
        reads=[],
        writes=["gbuf_albedo", "gbuf_normal", "gbuf_depth"]
    ))
    rg.add_pass(RenderPass(
        name="SSAO",
        reads=["gbuf_depth", "gbuf_normal"],
        writes=["ssao_texture"]
    ))
    rg.add_pass(RenderPass(
        name="Lighting",
        reads=["gbuf_albedo", "gbuf_normal", "gbuf_depth",
               "shadow_map", "ssao_texture"],
        writes=["hdr_color"]
    ))
    rg.add_pass(RenderPass(
        name="Bloom",
        reads=["hdr_color"],
        writes=["bloom_texture"]
    ))
    rg.add_pass(RenderPass(
        name="Tone Map",
        reads=["hdr_color", "bloom_texture"],
        writes=["ldr_output"]
    ))
    rg.add_pass(RenderPass(
        name="UI Overlay",
        reads=["ldr_output"],
        writes=["swapchain"]
    ))

    return rg


def demo_command_buffer():
    """Demonstrate command buffer recording for one render pass."""
    print("\n  --- Command Buffer Demo ---")

    # Create resources
    depth_rt = GPUResource("depth_buffer", 1024, 1024, 1)
    shadow_pso = PipelineState(
        name="shadow_depth",
        vertex_shader="shadow_vs",
        fragment_shader="shadow_fs",
        depth_test=True,
        depth_write=True,
        blend_enabled=False,
        cull_mode="front"  # Front-face culling reduces shadow acne
    )

    # Record commands
    cmd = CommandBuffer("shadow_pass")
    cmd.begin()
    cmd.barrier("depth_buffer", ResourceState.UNDEFINED, ResourceState.DEPTH_STENCIL)
    cmd.set_render_target("depth_buffer")
    cmd.clear("depth_buffer", (1.0, 0, 0, 0))
    cmd.bind_pipeline(shadow_pso)
    cmd.draw(36000, "scene geometry for shadow")
    cmd.barrier("depth_buffer", ResourceState.DEPTH_STENCIL, ResourceState.SHADER_READ)
    cmd.end()

    cmd.print_commands()

    # Simulate submission with fence
    fence = Fence("frame_complete")
    print(f"\n  Submitting command buffer '{cmd.name}' to GPU queue...")
    print(f"  Fence '{fence.name}' will be signaled on completion.")
    fence.signal()  # Simulated GPU completion
    fence.wait()
    print(f"  Fence '{fence.name}' signaled -- frame complete.")


def demo_render_graph():
    """Build, compile, and visualize a render graph."""
    print("\n  --- Render Graph Demo ---")

    rg = build_demo_frame()

    # Compile (topological sort)
    ordered = rg.compile()
    print("\n  Execution order (topological sort):")
    for i, p in enumerate(ordered):
        reads = ', '.join(p.reads) if p.reads else '(none)'
        writes = ', '.join(p.writes) if p.writes else '(none)'
        print(f"    [{i}] {p.name:12s}  reads: {reads}")
        print(f"        {'':12s}  writes: {writes}")

    # Compute barriers
    barriers = rg.compute_barriers(ordered)
    print("\n  Auto-inserted barriers:")
    total_barriers = 0
    for entry in barriers:
        if entry['barriers']:
            for b in entry['barriers']:
                print(f"    Before '{entry['pass']}': "
                      f"{b['resource']} {b['from'].name} -> {b['to'].name}")
                total_barriers += 1
    print(f"  Total barriers: {total_barriers}")

    # Compute lifetimes
    lifetimes = rg.compute_lifetimes(ordered)
    print("\n  Resource lifetimes (pass index range):")
    for name, (first, last) in sorted(lifetimes.items(), key=lambda x: x[1][0]):
        bar = '.' * first + '#' * (last - first + 1) + '.' * (len(ordered) - last - 1)
        print(f"    {name:20s} [{first}-{last}]  |{bar}|")

    return ordered, barriers, lifetimes


def demo_visualization(ordered, barriers, lifetimes):
    """Visualize the render graph structure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Modern Graphics API: Render Graph Analysis",
                 fontsize=14, fontweight='bold')

    # --- Top: Resource lifetime chart ---
    resources_sorted = sorted(lifetimes.items(), key=lambda x: x[1][0])
    y_positions = {}
    colors_map = plt.cm.Set3(np.linspace(0, 1, len(resources_sorted)))

    for i, (name, (first, last)) in enumerate(resources_sorted):
        y_positions[name] = i
        ax1.barh(i, last - first + 1, left=first, height=0.6,
                 color=colors_map[i], edgecolor='black', linewidth=0.5)
        ax1.text(first + (last - first + 1) / 2, i, name,
                 ha='center', va='center', fontsize=7, fontweight='bold')

    ax1.set_yticks(range(len(resources_sorted)))
    ax1.set_yticklabels([n for n, _ in resources_sorted], fontsize=8)
    ax1.set_xticks(range(len(ordered)))
    ax1.set_xticklabels([p.name for p in ordered], fontsize=8, rotation=30)
    ax1.set_xlabel("Render Pass")
    ax1.set_title("Resource Lifetimes (aliasable resources can overlap)")
    ax1.grid(axis='x', alpha=0.3)

    # --- Bottom: Barrier count per pass ---
    pass_names = [e['pass'] for e in barriers]
    barrier_counts = [len(e['barriers']) for e in barriers]
    bar_colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in barrier_counts]

    ax2.bar(range(len(pass_names)), barrier_counts, color=bar_colors,
            edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(pass_names)))
    ax2.set_xticklabels(pass_names, fontsize=9, rotation=30)
    ax2.set_ylabel("Barrier Count")
    ax2.set_title("Auto-Inserted Barriers per Pass (red = barriers needed)")
    ax2.grid(axis='y', alpha=0.3)

    for i, count in enumerate(barrier_counts):
        if count > 0:
            ax2.text(i, count + 0.1, str(count), ha='center', fontsize=9,
                     fontweight='bold')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_16_render_graph.png", dpi=100)
    plt.show()


def demo_api_comparison():
    """Visualize the complexity difference between old and new APIs."""
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = [
        'Lines of code\n(triangle)',
        'State\nvalidation',
        'Multi-thread\nrecording',
        'Memory\ncontrol',
        'Barrier\nmanagement',
        'Pipeline\nstate objects'
    ]

    # Relative complexity (higher = more application responsibility)
    opengl = [1, 0.2, 0.1, 0.1, 0, 0.2]
    vulkan = [5, 0.8, 0.9, 0.95, 1.0, 0.9]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, opengl, width, label='OpenGL (driver-managed)',
           color='#3498db', alpha=0.8)
    ax.bar(x + width / 2, vulkan, width, label='Vulkan (app-managed)',
           color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Application Responsibility (normalized)')
    ax.set_title('OpenGL vs Vulkan: Who Does the Work?', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.5, -0.18,
            "Vulkan trades simplicity for control -- more code, but predictable performance.",
            transform=ax.transAxes, ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig("/opt/projects/01_Personal/03_Study/examples/Computer_Graphics/"
                "output_16_api_comparison.png", dpi=100)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Modern Graphics APIs Simulation")
    print("=" * 60)

    print("\n[1/3] Command buffer recording...")
    demo_command_buffer()

    print("\n[2/3] Render graph compilation and analysis...")
    ordered, barriers, lifetimes = demo_render_graph()

    print("\n[3/3] Visualization...")
    demo_visualization(ordered, barriers, lifetimes)
    demo_api_comparison()

    print("\nDone!")


if __name__ == "__main__":
    main()

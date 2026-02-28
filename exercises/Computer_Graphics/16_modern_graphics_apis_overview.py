"""
Exercises for Lesson 16: Modern Graphics APIs Overview
Topic: Computer_Graphics
Solutions to practice problems from the lesson.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Exercise 1 -- API Comparison Pseudocode
# ---------------------------------------------------------------------------

def exercise_1():
    """
    Choose a simple rendering task (draw a textured, lit cube).
    Write pseudocode for OpenGL 3.3, Vulkan, and WebGPU.
    Count the number of API calls and objects created in each.
    """
    print("Exercise 1: API Comparison -- Draw a Textured Lit Cube")
    print()

    # We represent each API's workflow as a list of steps.
    # Each step is a (category, description) tuple.

    opengl_steps = [
        # --- Setup ---
        ("setup", "glutInit / glfwInit + create window"),
        ("setup", "glewInit / load GL function pointers"),
        # --- Shader ---
        ("shader", "glCreateShader(GL_VERTEX_SHADER)"),
        ("shader", "glShaderSource(vs, vert_code)"),
        ("shader", "glCompileShader(vs)"),
        ("shader", "glCreateShader(GL_FRAGMENT_SHADER)"),
        ("shader", "glShaderSource(fs, frag_code)"),
        ("shader", "glCompileShader(fs)"),
        ("shader", "glCreateProgram()"),
        ("shader", "glAttachShader(program, vs)"),
        ("shader", "glAttachShader(program, fs)"),
        ("shader", "glLinkProgram(program)"),
        # --- Geometry ---
        ("geometry", "glGenVertexArrays(1, &vao)"),
        ("geometry", "glBindVertexArray(vao)"),
        ("geometry", "glGenBuffers(1, &vbo)"),
        ("geometry", "glBindBuffer(GL_ARRAY_BUFFER, vbo)"),
        ("geometry", "glBufferData(GL_ARRAY_BUFFER, cube_vertices, GL_STATIC_DRAW)"),
        ("geometry", "glVertexAttribPointer(0, ...) // position"),
        ("geometry", "glEnableVertexAttribArray(0)"),
        ("geometry", "glVertexAttribPointer(1, ...) // normal"),
        ("geometry", "glEnableVertexAttribArray(1)"),
        ("geometry", "glVertexAttribPointer(2, ...) // texcoord"),
        ("geometry", "glEnableVertexAttribArray(2)"),
        ("geometry", "glGenBuffers(1, &ebo)"),
        ("geometry", "glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)"),
        ("geometry", "glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)"),
        # --- Texture ---
        ("texture", "glGenTextures(1, &tex)"),
        ("texture", "glBindTexture(GL_TEXTURE_2D, tex)"),
        ("texture", "glTexImage2D(GL_TEXTURE_2D, ..., pixel_data)"),
        ("texture", "glGenerateMipmap(GL_TEXTURE_2D)"),
        ("texture", "glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, ...)"),
        ("texture", "glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, ...)"),
        # --- Per frame ---
        ("frame", "glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)"),
        ("frame", "glUseProgram(program)"),
        ("frame", "glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &mvp)"),
        ("frame", "glUniform3fv(light_pos_loc, 1, &light_pos)"),
        ("frame", "glActiveTexture(GL_TEXTURE0)"),
        ("frame", "glBindTexture(GL_TEXTURE_2D, tex)"),
        ("frame", "glBindVertexArray(vao)"),
        ("frame", "glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0)"),
        ("frame", "glfwSwapBuffers(window)"),
    ]

    vulkan_steps = [
        # --- Instance + Device ---
        ("setup", "vkCreateInstance(&instanceInfo, &instance)"),
        ("setup", "vkEnumeratePhysicalDevices(instance, &count, ...)"),
        ("setup", "vkGetPhysicalDeviceQueueFamilyProperties(...)"),
        ("setup", "vkCreateDevice(physicalDevice, &deviceInfo, &device)"),
        ("setup", "vkGetDeviceQueue(device, graphicsFamily, 0, &queue)"),
        # --- Surface + Swapchain ---
        ("setup", "vkCreateSurfaceKHR(instance, &surfaceInfo, &surface)"),
        ("setup", "vkGetPhysicalDeviceSurfaceCapabilitiesKHR(...)"),
        ("setup", "vkCreateSwapchainKHR(device, &swapInfo, &swapchain)"),
        ("setup", "vkGetSwapchainImagesKHR(device, swapchain, ...)"),
        ("setup", "create VkImageView for each swapchain image (x3)"),
        # --- Render Pass ---
        ("renderpass", "vkCreateRenderPass(device, &rpInfo, &renderPass)"),
        ("renderpass", "create VkFramebuffer for each swapchain image (x3)"),
        # --- Pipeline ---
        ("pipeline", "load SPIR-V vertex shader bytes"),
        ("pipeline", "vkCreateShaderModule(device, &vsInfo, &vertModule)"),
        ("pipeline", "load SPIR-V fragment shader bytes"),
        ("pipeline", "vkCreateShaderModule(device, &fsInfo, &fragModule)"),
        ("pipeline", "define VkPipelineVertexInputStateCreateInfo"),
        ("pipeline", "define VkPipelineInputAssemblyStateCreateInfo"),
        ("pipeline", "define VkPipelineViewportStateCreateInfo"),
        ("pipeline", "define VkPipelineRasterizationStateCreateInfo"),
        ("pipeline", "define VkPipelineMultisampleStateCreateInfo"),
        ("pipeline", "define VkPipelineDepthStencilStateCreateInfo"),
        ("pipeline", "define VkPipelineColorBlendStateCreateInfo"),
        ("pipeline", "vkCreateDescriptorSetLayout(device, ..., &setLayout)"),
        ("pipeline", "vkCreatePipelineLayout(device, ..., &pipelineLayout)"),
        ("pipeline", "vkCreateGraphicsPipelines(device, cache, 1, &pipeInfo, &pipeline)"),
        # --- Buffers ---
        ("buffer", "vkCreateBuffer(device, vertex staging buffer)"),
        ("buffer", "vkAllocateMemory(device, HOST_VISIBLE, &stagingMem)"),
        ("buffer", "vkMapMemory + memcpy cube vertices + vkUnmapMemory"),
        ("buffer", "vkCreateBuffer(device, vertex device-local buffer)"),
        ("buffer", "vkAllocateMemory(device, DEVICE_LOCAL, &vertMem)"),
        ("buffer", "record + submit copy command (staging -> device)"),
        ("buffer", "vkCreateBuffer(device, index staging buffer)"),
        ("buffer", "vkAllocateMemory + copy indices + transfer"),
        ("buffer", "vkCreateBuffer(device, uniform buffer)"),
        ("buffer", "vkAllocateMemory(device, HOST_VISIBLE for uniform)"),
        # --- Texture ---
        ("texture", "vkCreateImage(device, &imageInfo, &textureImage)"),
        ("texture", "vkAllocateMemory(device, DEVICE_LOCAL, &texMem)"),
        ("texture", "vkBindImageMemory(device, textureImage, texMem, 0)"),
        ("texture", "create staging buffer, copy pixel data"),
        ("texture", "transition image layout (UNDEFINED -> TRANSFER_DST)"),
        ("texture", "vkCmdCopyBufferToImage(cmd, staging, textureImage)"),
        ("texture", "transition image layout (TRANSFER_DST -> SHADER_READ)"),
        ("texture", "vkCreateImageView(device, &viewInfo, &texView)"),
        ("texture", "vkCreateSampler(device, &samplerInfo, &sampler)"),
        # --- Descriptors ---
        ("descriptor", "vkCreateDescriptorPool(device, &poolInfo, &pool)"),
        ("descriptor", "vkAllocateDescriptorSets(device, &allocInfo, &set)"),
        ("descriptor", "vkUpdateDescriptorSets(device, writes for UBO+texture)"),
        # --- Sync ---
        ("sync", "vkCreateSemaphore(device, ..., &imageAvailable)"),
        ("sync", "vkCreateSemaphore(device, ..., &renderFinished)"),
        ("sync", "vkCreateFence(device, ..., &inFlightFence)"),
        # --- Command ---
        ("command", "vkCreateCommandPool(device, &poolInfo, &cmdPool)"),
        ("command", "vkAllocateCommandBuffers(device, ..., &cmdBuffer)"),
        # --- Per frame ---
        ("frame", "vkWaitForFences(device, 1, &fence, ...)"),
        ("frame", "vkResetFences(device, 1, &fence)"),
        ("frame", "vkAcquireNextImageKHR(device, swapchain, ..., &imgIdx)"),
        ("frame", "vkResetCommandBuffer(cmdBuffer, 0)"),
        ("frame", "vkBeginCommandBuffer(cmdBuffer, &beginInfo)"),
        ("frame", "vkCmdBeginRenderPass(cmdBuffer, &rpBeginInfo)"),
        ("frame", "vkCmdBindPipeline(cmdBuffer, GRAPHICS, pipeline)"),
        ("frame", "vkCmdBindDescriptorSets(cmdBuffer, ..., &set)"),
        ("frame", "vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &vertBuf, ...)"),
        ("frame", "vkCmdBindIndexBuffer(cmdBuffer, indexBuf, 0, UINT32)"),
        ("frame", "update uniform buffer (map, write MVP + light, unmap)"),
        ("frame", "vkCmdDrawIndexed(cmdBuffer, 36, 1, 0, 0, 0)"),
        ("frame", "vkCmdEndRenderPass(cmdBuffer)"),
        ("frame", "vkEndCommandBuffer(cmdBuffer)"),
        ("frame", "vkQueueSubmit(queue, 1, &submitInfo, fence)"),
        ("frame", "vkQueuePresentKHR(queue, &presentInfo)"),
    ]

    webgpu_steps = [
        # --- Setup ---
        ("setup", "navigator.gpu.requestAdapter()"),
        ("setup", "adapter.requestDevice()"),
        ("setup", "canvas.getContext('webgpu')"),
        ("setup", "context.configure({ device, format })"),
        # --- Shader ---
        ("shader", "device.createShaderModule({ code: wgslCode })"),
        # --- Pipeline ---
        ("pipeline", "device.createRenderPipeline({ vertex, fragment, primitive, "
                      "depthStencil })"),
        # --- Buffers ---
        ("buffer", "device.createBuffer({ size, usage: VERTEX, mappedAtCreation })"),
        ("buffer", "write cube vertices to mapped buffer"),
        ("buffer", "device.createBuffer({ size, usage: INDEX, mappedAtCreation })"),
        ("buffer", "write indices to mapped buffer"),
        ("buffer", "device.createBuffer({ size, usage: UNIFORM | COPY_DST })"),
        # --- Texture ---
        ("texture", "device.createTexture({ size, format, usage })"),
        ("texture", "device.queue.writeTexture(texture, pixelData, ...)"),
        ("texture", "device.createSampler({ magFilter, minFilter, mipmapFilter })"),
        # --- Bind Group ---
        ("bind", "device.createBindGroup({ layout, entries: [ubo, texture, sampler] })"),
        # --- Depth ---
        ("depth", "device.createTexture({ format: depth24plus, usage: RENDER_ATTACHMENT })"),
        # --- Per frame ---
        ("frame", "device.queue.writeBuffer(uniformBuf, 0, mvpData)"),
        ("frame", "device.createCommandEncoder()"),
        ("frame", "encoder.beginRenderPass(renderPassDescriptor)"),
        ("frame", "pass.setPipeline(pipeline)"),
        ("frame", "pass.setBindGroup(0, bindGroup)"),
        ("frame", "pass.setVertexBuffer(0, vertexBuffer)"),
        ("frame", "pass.setIndexBuffer(indexBuffer, 'uint32')"),
        ("frame", "pass.drawIndexed(36)"),
        ("frame", "pass.end()"),
        ("frame", "device.queue.submit([encoder.finish()])"),
    ]

    apis = [
        ("OpenGL 3.3", opengl_steps),
        ("Vulkan", vulkan_steps),
        ("WebGPU", webgpu_steps),
    ]

    for name, steps in apis:
        categories = defaultdict(int)
        for cat, _ in steps:
            categories[cat] += 1
        total = len(steps)
        print(f"  {name}:")
        print(f"    Total API calls/steps: {total}")
        for cat in ['setup', 'shader', 'renderpass', 'pipeline', 'buffer',
                     'texture', 'descriptor', 'bind', 'sync', 'command',
                     'depth', 'frame', 'geometry']:
            if categories[cat] > 0:
                print(f"      {cat:12s}: {categories[cat]:3d}")
        print()

    # Comparison table
    print("  Summary Comparison:")
    print(f"  {'API':12s} | {'Total Steps':>11s} | {'Setup':>5s} | {'Per-Frame':>9s} | "
          f"{'Objects':>7s}")
    print(f"  {'-'*12:s}-+-{'-'*11:s}-+-{'-'*5:s}-+-{'-'*9:s}-+-{'-'*7:s}")
    for name, steps in apis:
        cats = defaultdict(int)
        for cat, _ in steps:
            cats[cat] += 1
        non_frame = sum(v for k, v in cats.items() if k != 'frame')
        print(f"  {name:12s} | {len(steps):11d} | {non_frame:5d} | "
              f"{cats['frame']:9d} | ~{non_frame // 2:5d}")

    print()
    print("  Key insight: Vulkan requires ~3-4x more setup than OpenGL, but gives")
    print("  explicit control over memory, synchronization, and multi-threading.")
    print("  WebGPU provides a good middle ground with modern features and safety.")


# ---------------------------------------------------------------------------
# Exercise 2 -- Render Graph with Memory Aliasing
# ---------------------------------------------------------------------------

def exercise_2():
    """
    Extend the render graph implementation to support memory aliasing:
    resources that do not overlap in the execution timeline share the
    same memory allocation.  Print total memory with and without aliasing.
    """
    @dataclass
    class Resource:
        name: str
        size_mb: float  # memory footprint

    @dataclass
    class RenderPass:
        name: str
        reads: Set[str] = field(default_factory=set)
        writes: Set[str] = field(default_factory=set)

        def read(self, name):
            self.reads.add(name)
            return self

        def write(self, name):
            self.writes.add(name)
            return self

    class RenderGraph:
        def __init__(self):
            self.passes: List[RenderPass] = []
            self.resources: Dict[str, Resource] = {}
            self.final_outputs: Set[str] = set()

        def add_resource(self, name, size_mb):
            self.resources[name] = Resource(name, size_mb)

        def add_pass(self, name) -> RenderPass:
            p = RenderPass(name)
            self.passes.append(p)
            return p

        def set_output(self, name):
            self.final_outputs.add(name)

        def compile(self):
            """Topological sort with dead-pass culling."""
            writer_of = {}
            for p in self.passes:
                for w in p.writes:
                    writer_of[w] = p

            # Reverse reachability from final outputs
            needed = set()
            stack = [p for p in self.passes if p.writes & self.final_outputs]
            while stack:
                p = stack.pop()
                if p.name in needed:
                    continue
                needed.add(p.name)
                for r in p.reads:
                    if r in writer_of and writer_of[r].name not in needed:
                        stack.append(writer_of[r])

            active = [p for p in self.passes if p.name in needed]

            # Topological sort (Kahn's algorithm)
            in_deg = {p.name: 0 for p in active}
            edges = defaultdict(list)
            for p in active:
                for r in p.reads:
                    if r in writer_of and writer_of[r].name in needed:
                        dep = writer_of[r].name
                        if dep != p.name:
                            edges[dep].append(p.name)
                            in_deg[p.name] += 1

            queue = [p for p in active if in_deg[p.name] == 0]
            ordered = []
            while queue:
                p = queue.pop(0)
                ordered.append(p)
                for nb in edges[p.name]:
                    in_deg[nb] -= 1
                    if in_deg[nb] == 0:
                        queue.append(next(x for x in active if x.name == nb))

            return ordered

        def compute_lifetimes(self, ordered):
            """
            Determine the [first_write, last_read] interval for each resource.
            The interval is measured in pass indices within the ordered list.
            """
            pass_idx = {p.name: i for i, p in enumerate(ordered)}
            lifetimes = {}  # resource -> (first_used, last_used)

            for p in ordered:
                idx = pass_idx[p.name]
                for r in p.writes:
                    if r not in lifetimes:
                        lifetimes[r] = [idx, idx]
                    else:
                        lifetimes[r][0] = min(lifetimes[r][0], idx)
                        lifetimes[r][1] = max(lifetimes[r][1], idx)
                for r in p.reads:
                    if r not in lifetimes:
                        lifetimes[r] = [idx, idx]
                    else:
                        lifetimes[r][1] = max(lifetimes[r][1], idx)

            return lifetimes

        def alias_memory(self, ordered):
            """
            Greedy memory aliasing: assign resources to memory slots.
            Non-overlapping resources can share the same slot.
            Returns (slots, total_aliased_mb, total_naive_mb).
            """
            lifetimes = self.compute_lifetimes(ordered)

            # Sort resources by size (descending) for better packing
            res_list = sorted(
                [(name, lifetimes[name]) for name in lifetimes
                 if name in self.resources],
                key=lambda x: -self.resources[x[0]].size_mb
            )

            # Each slot: (size_mb, list of (resource, start, end))
            slots = []

            for res_name, (start, end) in res_list:
                size = self.resources[res_name].size_mb
                placed = False

                # Try to fit into an existing slot (no overlap)
                for slot in slots:
                    slot_size, occupants = slot
                    conflict = False
                    for _, occ_start, occ_end in occupants:
                        if not (end < occ_start or start > occ_end):
                            conflict = True
                            break
                    if not conflict:
                        # Place here; slot size is the max of its occupants
                        occupants.append((res_name, start, end))
                        if size > slot[0]:
                            slot[0] = size
                        placed = True
                        break

                if not placed:
                    slots.append([size, [(res_name, start, end)]])

            total_naive = sum(self.resources[r].size_mb
                              for r in lifetimes if r in self.resources)
            total_aliased = sum(s[0] for s in slots)

            return slots, total_aliased, total_naive

    # --- Build a realistic deferred rendering frame graph ---
    graph = RenderGraph()

    # Resources with realistic sizes (1920x1080 @ FP16/RGBA8)
    resources = {
        'shadow_map':   4.0,    # 2048x2048 depth
        'gbuf_albedo':  8.0,    # 1920x1080 RGBA8
        'gbuf_normal':  16.0,   # 1920x1080 RGBA16F
        'gbuf_depth':   8.0,    # 1920x1080 D32F
        'ao_tex':       4.0,    # 1920x1080 R8 (after blur)
        'ao_raw':       4.0,    # 1920x1080 R8 (before blur)
        'hdr_color':    16.0,   # 1920x1080 RGBA16F
        'bloom_bright': 16.0,   # 1920x1080 RGBA16F
        'bloom_blur':   4.0,    # Downsampled blur chain
        'ldr_color':    8.0,    # 1920x1080 RGBA8 (final)
    }
    for name, size in resources.items():
        graph.add_resource(name, size)

    # Passes
    shadow = graph.add_pass("ShadowMap")
    shadow.write("shadow_map")

    gbuffer = graph.add_pass("GBuffer")
    gbuffer.write("gbuf_albedo").write("gbuf_normal").write("gbuf_depth")

    ssao = graph.add_pass("SSAO")
    ssao.read("gbuf_normal").read("gbuf_depth").write("ao_raw")

    ssao_blur = graph.add_pass("SSAO_Blur")
    ssao_blur.read("ao_raw").write("ao_tex")

    lighting = graph.add_pass("Lighting")
    (lighting.read("gbuf_albedo").read("gbuf_normal").read("gbuf_depth")
     .read("shadow_map").read("ao_tex").write("hdr_color"))

    bloom_extract = graph.add_pass("BloomExtract")
    bloom_extract.read("hdr_color").write("bloom_bright")

    bloom_blur = graph.add_pass("BloomBlur")
    bloom_blur.read("bloom_bright").write("bloom_blur")

    tonemap = graph.add_pass("ToneMap")
    tonemap.read("hdr_color").read("bloom_blur").write("ldr_color")

    # Debug pass (should be culled)
    debug = graph.add_pass("DebugNormals")
    debug.read("gbuf_normal").write("debug_output")
    graph.add_resource("debug_output", 8.0)

    graph.set_output("ldr_color")

    # Compile and analyze
    ordered = graph.compile()

    print("Exercise 2: Render Graph with Memory Aliasing")
    print()
    print(f"  Passes declared: {len(graph.passes)}")
    print(f"  Passes after culling: {len(ordered)}")
    print(f"  Execution order: {' -> '.join(p.name for p in ordered)}")
    print()

    # Lifetime analysis
    lifetimes = graph.compute_lifetimes(ordered)
    pass_names = [p.name for p in ordered]
    print("  Resource lifetimes (pass index range):")
    for res_name in sorted(lifetimes.keys()):
        start, end = lifetimes[res_name]
        size = graph.resources.get(res_name, Resource(res_name, 0)).size_mb
        bar = '.' * start + '#' * (end - start + 1) + '.' * (len(ordered) - end - 1)
        print(f"    {res_name:15s} ({size:5.1f} MB): [{bar}] "
              f"passes {start}-{end}")

    # Memory aliasing
    slots, total_aliased, total_naive = graph.alias_memory(ordered)
    print()
    print(f"  Memory without aliasing: {total_naive:.1f} MB")
    print(f"  Memory with aliasing:    {total_aliased:.1f} MB")
    print(f"  Savings:                 {total_naive - total_aliased:.1f} MB "
          f"({(1 - total_aliased / total_naive) * 100:.1f}%)")
    print()

    print("  Memory slots (aliased resources share a slot):")
    for i, (size, occupants) in enumerate(slots):
        occ_str = ', '.join(f'{name}[{s}-{e}]' for name, s, e in occupants)
        print(f"    Slot {i}: {size:5.1f} MB -- {occ_str}")


# ---------------------------------------------------------------------------
# Exercise 3 -- Synchronization Design
# ---------------------------------------------------------------------------

def exercise_3():
    """
    Design the fence and semaphore setup for a triple-buffered Vulkan
    application with async compute.  Draw a timeline showing which fences
    and semaphores are signaled/waited on each frame.
    """
    print("Exercise 3: Synchronization Design (Triple-Buffered + Async Compute)")
    print()

    NUM_FRAMES = 3  # Triple buffering
    TOTAL_FRAMES = 6  # Simulate 6 frames

    @dataclass
    class SyncPrimitive:
        name: str
        kind: str        # 'fence' or 'semaphore'
        signaled: bool = False

    @dataclass
    class FrameResources:
        """Per-frame synchronization primitives."""
        index: int
        # Fences (CPU-GPU): CPU waits before reusing this frame's resources
        in_flight_fence: SyncPrimitive = None
        # Semaphores (GPU-GPU):
        image_available: SyncPrimitive = None    # swapchain -> graphics
        render_finished: SyncPrimitive = None    # graphics -> present
        compute_finished: SyncPrimitive = None   # compute -> graphics (async)

        def __post_init__(self):
            i = self.index
            self.in_flight_fence = SyncPrimitive(
                f"fence_{i}", "fence", signaled=True)  # Start signaled
            self.image_available = SyncPrimitive(
                f"img_avail_{i}", "semaphore")
            self.render_finished = SyncPrimitive(
                f"render_done_{i}", "semaphore")
            self.compute_finished = SyncPrimitive(
                f"compute_done_{i}", "semaphore")

    frames = [FrameResources(i) for i in range(NUM_FRAMES)]
    timeline = []

    print("  Primitives per frame set:")
    print("    - in_flight_fence:   CPU waits before reusing command buffers")
    print("    - image_available:   signaled when swapchain image is ready")
    print("    - render_finished:   signaled when graphics queue completes")
    print("    - compute_finished:  signaled when async compute completes")
    print()

    for frame_num in range(TOTAL_FRAMES):
        fi = frame_num % NUM_FRAMES
        fr = frames[fi]
        events = []

        # Step 1: CPU waits on fence (ensure frame fi resources are free)
        events.append(f"CPU: WaitFence({fr.in_flight_fence.name})")
        fr.in_flight_fence.signaled = False

        # Step 2: Acquire swapchain image
        events.append(f"CPU: AcquireImage -> signals {fr.image_available.name}")

        # Step 3: Record and submit async compute
        events.append(
            f"CPU: Submit compute queue "
            f"-> signals {fr.compute_finished.name}")

        # Step 4: Record and submit graphics
        events.append(
            f"CPU: Submit graphics queue "
            f"(waits {fr.image_available.name}, {fr.compute_finished.name}) "
            f"-> signals {fr.render_finished.name}, {fr.in_flight_fence.name}")
        fr.in_flight_fence.signaled = True

        # Step 5: Present
        events.append(
            f"CPU: Present (waits {fr.render_finished.name})")

        timeline.append((frame_num, fi, events))

    # Print timeline
    print("  Frame Timeline:")
    print("  " + "=" * 75)
    for frame_num, fi, events in timeline:
        print(f"  Frame {frame_num} (resource set {fi}):")
        for ev in events:
            print(f"    {ev}")
        print()

    # ASCII timeline diagram
    print("  Visual Timeline (time flows right ->):")
    print()
    cols = 12
    header = "  " + "".join(f"  Frame {i:<{cols - 8}d}" for i in range(TOTAL_FRAMES))
    print(header)
    print("  " + "-" * (cols * TOTAL_FRAMES))

    rows = [
        ("CPU record", lambda fn: f"set{fn % NUM_FRAMES}"),
        ("Compute Q", lambda fn: f"async"),
        ("Graphics Q", lambda fn: f"draw"),
        ("Present",   lambda fn: f"show"),
    ]

    for label, fn in rows:
        line = f"  {label:12s}"
        for frame_num in range(TOTAL_FRAMES):
            cell = fn(frame_num)
            line += f"[{cell:^{cols - 2}s}]"
        print(line)

    print()
    print("  Key synchronization rules:")
    print("    1. Never reuse frame N's resources until fence_N is signaled")
    print("    2. Graphics waits on image_available before writing to swapchain image")
    print("    3. Graphics waits on compute_finished if it depends on compute results")
    print("    4. Present waits on render_finished before displaying")
    print("    5. Triple buffering keeps GPU fed: CPU works on frame N+2 while")
    print("       GPU executes frame N+1 and displays frame N")


# ---------------------------------------------------------------------------
# Exercise 4 -- Pipeline State Explosion
# ---------------------------------------------------------------------------

def exercise_4():
    """
    A material system supports 3 vertex formats, 2 blend modes, 4 shader
    variants, and 3 render pass configurations.  Calculate total PSOs needed.
    Discuss pipeline caches and lazy creation to manage this.
    """
    print("Exercise 4: Pipeline State Explosion")
    print()

    # Define the dimensions of variation
    vertex_formats = ["StaticMesh", "SkinnedMesh", "Particle"]
    blend_modes = ["Opaque", "AlphaBlend"]
    shader_variants = ["Forward", "Deferred", "Shadow", "DepthOnly"]
    render_passes = ["MainColor", "ShadowMap", "PostProcess"]

    n_vf = len(vertex_formats)
    n_bm = len(blend_modes)
    n_sv = len(shader_variants)
    n_rp = len(render_passes)
    total_naive = n_vf * n_bm * n_sv * n_rp

    print(f"  Vertex formats:   {n_vf} ({', '.join(vertex_formats)})")
    print(f"  Blend modes:      {n_bm} ({', '.join(blend_modes)})")
    print(f"  Shader variants:  {n_sv} ({', '.join(shader_variants)})")
    print(f"  Render passes:    {n_rp} ({', '.join(render_passes)})")
    print()
    print(f"  Naive total PSOs: {n_vf} x {n_bm} x {n_sv} x {n_rp} = {total_naive}")
    print()

    # Not all combinations are valid.  Prune invalid ones.
    valid_psos = []
    invalid_psos = []

    for vf in vertex_formats:
        for bm in blend_modes:
            for sv in shader_variants:
                for rp in render_passes:
                    key = (vf, bm, sv, rp)
                    valid = True
                    reason = ""

                    # Shadow and DepthOnly don't use AlphaBlend (they only write depth)
                    if sv in ("Shadow", "DepthOnly") and bm == "AlphaBlend":
                        valid = False
                        reason = "depth-only passes ignore blend mode"

                    # Deferred doesn't support alpha blend (G-buffer is opaque)
                    if sv == "Deferred" and bm == "AlphaBlend":
                        valid = False
                        reason = "deferred G-buffer only supports opaque"

                    # Shadow variant only used with ShadowMap render pass
                    if sv == "Shadow" and rp != "ShadowMap":
                        valid = False
                        reason = "shadow shader only used in shadow pass"

                    # PostProcess render pass uses only Forward variant
                    if rp == "PostProcess" and sv not in ("Forward",):
                        valid = False
                        reason = "post-process only uses forward variant"

                    # Particles don't use deferred (typically forward-rendered)
                    if vf == "Particle" and sv == "Deferred":
                        valid = False
                        reason = "particles are forward-rendered"

                    if valid:
                        valid_psos.append(key)
                    else:
                        invalid_psos.append((key, reason))

    print(f"  After pruning invalid combinations: {len(valid_psos)} valid PSOs")
    print(f"  Eliminated: {len(invalid_psos)} invalid combinations")
    print()

    # Show some examples of pruned combinations
    print("  Sample invalid combinations (first 8):")
    for (vf, bm, sv, rp), reason in invalid_psos[:8]:
        print(f"    {vf}/{bm}/{sv}/{rp} -- {reason}")
    print()

    # Lazy creation simulation
    print("  Management Strategies:")
    print()

    # Strategy 1: Pipeline cache
    print("  1. Pipeline Cache:")
    print("     - Serialize compiled PSOs to disk after first creation")
    print("     - On subsequent launches, load from cache (avoids shader compilation)")
    print(f"     - Cache file stores all {len(valid_psos)} compiled PSOs (~2-10 MB)")
    print("     - Vulkan: VkPipelineCache; DX12: ID3D12PipelineState serialization")
    print()

    # Strategy 2: Lazy creation
    psos_created = set()
    frame_requests = [
        # Frame 1: basic scene (subset of PSOs actually needed)
        [("StaticMesh", "Opaque", "Shadow", "ShadowMap"),
         ("StaticMesh", "Opaque", "Deferred", "MainColor"),
         ("StaticMesh", "Opaque", "Forward", "PostProcess")],
        # Frame 2: add skinned character
        [("SkinnedMesh", "Opaque", "Shadow", "ShadowMap"),
         ("SkinnedMesh", "Opaque", "Deferred", "MainColor"),
         ("StaticMesh", "Opaque", "Deferred", "MainColor")],
        # Frame 3: add particles
        [("Particle", "AlphaBlend", "Forward", "MainColor"),
         ("StaticMesh", "Opaque", "Shadow", "ShadowMap")],
    ]

    print("  2. Lazy Creation (create PSOs on first use):")
    for i, requests in enumerate(frame_requests):
        new_this_frame = 0
        for pso_key in requests:
            if pso_key not in psos_created:
                psos_created.add(pso_key)
                new_this_frame += 1
        print(f"     Frame {i}: requested {len(requests)} PSOs, "
              f"created {new_this_frame} new (total cached: {len(psos_created)})")

    print()
    print(f"  After 3 frames: {len(psos_created)}/{len(valid_psos)} valid PSOs created")
    print("  Remaining PSOs created only when needed (e.g., new material/effect)")
    print()

    # Strategy 3: Async compilation
    print("  3. Async Compilation:")
    print("     - Compile PSOs on background threads during loading screen")
    print("     - If a PSO is needed before compilation finishes, use a fallback")
    print("       (e.g., simpler shader variant) until the real one is ready")
    print("     - Eliminates frame hitches from runtime shader compilation")


# ---------------------------------------------------------------------------
# Exercise 5 -- Bindless vs. Bound CPU Overhead
# ---------------------------------------------------------------------------

def exercise_5():
    """
    Compare the CPU overhead of drawing 1000 objects with:
    (a) per-object descriptor set binds
    (b) bindless rendering
    Estimate the number of API calls for each approach.
    """
    import time

    NUM_OBJECTS = 1000
    NUM_TEXTURES = 200   # Shared texture pool
    NUM_FRAMES = 100     # Frames to simulate

    print("Exercise 5: Bindless vs. Bound CPU Overhead")
    print(f"  Objects: {NUM_OBJECTS}, Textures: {NUM_TEXTURES}")
    print()

    # Assign random textures to objects
    rng = np.random.RandomState(42)
    object_textures = rng.randint(0, NUM_TEXTURES, size=NUM_OBJECTS)

    # --- Approach A: Per-object descriptor set binding ---
    # For each object: bind descriptor set, bind vertex buffer, draw
    # In Vulkan, this means:
    #   vkCmdBindDescriptorSets (1 per object, includes texture binding)
    #   vkCmdBindVertexBuffers  (1 per object, if different meshes)
    #   vkCmdDrawIndexed        (1 per object)
    #   Total: 3 * NUM_OBJECTS per frame

    bound_calls_per_frame = 0
    t0 = time.perf_counter()
    for _ in range(NUM_FRAMES):
        # Simulate sorting by texture to minimize state changes
        sorted_indices = np.argsort(object_textures)
        last_tex = -1
        binds = 0
        draws = 0

        for obj_idx in sorted_indices:
            tex = object_textures[obj_idx]
            if tex != last_tex:
                binds += 1  # vkCmdBindDescriptorSets
                last_tex = tex
            draws += 1      # vkCmdDrawIndexed (+ vertex bind)

        bound_calls_per_frame = binds + draws * 2  # bind desc + bind VB + draw
    bound_time = (time.perf_counter() - t0) / NUM_FRAMES

    # --- Approach B: Bindless rendering ---
    # All textures in one descriptor array. Each object stores a material_id.
    # Per object: push constant (material ID) + draw. Or even better:
    # Use indirect draw with per-instance material ID -> one draw call.
    #
    # Minimal: 1 bind descriptor set (once), 1 bind vertex buffer (once),
    #          1 vkCmdDrawIndexedIndirect for all objects.

    bindless_calls_per_frame = 0
    t0 = time.perf_counter()
    for _ in range(NUM_FRAMES):
        # Bindless: bind once, then indirect draw
        api_calls = 0
        api_calls += 1  # vkCmdBindDescriptorSets (one big set with all textures)
        api_calls += 1  # vkCmdBindVertexBuffers (single merged vertex buffer)
        api_calls += 1  # vkCmdBindIndexBuffer
        # Option A: one indirect draw for all 1000 objects
        api_calls += 1  # vkCmdDrawIndexedIndirect(cmd, indirectBuf, 0, 1000, stride)
        bindless_calls_per_frame = api_calls
    bindless_time = (time.perf_counter() - t0) / NUM_FRAMES

    # Sort-optimized bound (batch by texture)
    unique_textures = len(set(object_textures))

    print("  Approach A: Per-Object Descriptor Binding (texture-sorted)")
    print(f"    Unique textures used: {unique_textures}")
    print(f"    Descriptor set binds: {unique_textures} (once per texture change)")
    print(f"    Vertex buffer binds:  {NUM_OBJECTS}")
    print(f"    Draw calls:           {NUM_OBJECTS}")
    print(f"    Total API calls/frame: ~{bound_calls_per_frame}")
    print(f"    CPU simulation time:  {bound_time * 1000:.3f} ms/frame")
    print()

    print("  Approach B: Bindless Rendering (indirect draw)")
    print(f"    Descriptor set binds: 1 (all textures in one array)")
    print(f"    Vertex buffer binds:  1 (merged buffer)")
    print(f"    Draw calls:           1 (indirect, {NUM_OBJECTS} instances)")
    print(f"    Total API calls/frame: {bindless_calls_per_frame}")
    print(f"    CPU simulation time:  {bindless_time * 1000:.3f} ms/frame")
    print()

    reduction = bound_calls_per_frame / max(1, bindless_calls_per_frame)
    print(f"  API call reduction: {reduction:.0f}x fewer calls with bindless")
    print()

    print("  Analysis:")
    print(f"    Bound approach:    {bound_calls_per_frame:,d} API calls/frame")
    print(f"    Bindless approach: {bindless_calls_per_frame:,d} API calls/frame")
    print()
    print("  Why bindless wins:")
    print("    - Each API call has CPU overhead (validation, state tracking)")
    print("    - Driver command buffer recording scales with call count")
    print("    - Bindless batches everything into minimal GPU submissions")
    print("    - Indirect draw moves per-object decisions to GPU")
    print()
    print("  Bindless requirements:")
    print("    - GPU must support large descriptor arrays")
    print("    - Vulkan: VK_EXT_descriptor_indexing / Vulkan 1.2")
    print("    - DX12: unbounded descriptor heaps (Shader Model 6.6+)")
    print("    - WebGPU: not yet supported (limited binding model)")


# ---------------------------------------------------------------------------
# Exercise 6 -- WebGPU Compute Shader Design
# ---------------------------------------------------------------------------

def exercise_6():
    """
    Write a WebGPU compute shader (in WGSL) that doubles every element in a
    storage buffer.  Write the JavaScript code to create the device, buffer,
    pipeline, and dispatch the shader.  (Simulated in Python since we cannot
    run actual WebGPU here.)
    """
    print("Exercise 6: WebGPU Compute Shader Design")
    print()

    # --- WGSL Shader Code ---
    wgsl_shader = """
// WebGPU Compute Shader: double every element in a storage buffer
//
// @group(0) @binding(0): read-write storage buffer of f32 values
// Workgroup size: 64 invocations (a common choice for compute)

@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    // Guard against out-of-bounds access when buffer size is not
    // a multiple of the workgroup size.
    if (index < arrayLength(&data)) {
        data[index] = data[index] * 2.0;
    }
}
""".strip()

    # --- JavaScript Host Code ---
    js_code = """
// WebGPU JavaScript host code: create device, buffer, pipeline, dispatch

async function doubleBuffer() {
    // 1. Request adapter and device
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No GPU adapter found');
    const device = await adapter.requestDevice();

    // 2. Create the compute shader module
    const shaderModule = device.createShaderModule({
        code: `
            @group(0) @binding(0)
            var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let index = gid.x;
                if (index < arrayLength(&data)) {
                    data[index] = data[index] * 2.0;
                }
            }
        `
    });

    // 3. Create the compute pipeline
    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });

    // 4. Create input data
    const N = 1024;
    const inputData = new Float32Array(N);
    for (let i = 0; i < N; i++) {
        inputData[i] = i + 1;  // [1, 2, 3, ..., 1024]
    }

    // 5. Create GPU storage buffer (read-write, copy-src for readback)
    const gpuBuffer = device.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.STORAGE |
               GPUBufferUsage.COPY_DST |
               GPUBufferUsage.COPY_SRC,
        mappedAtCreation: false,
    });

    // 6. Upload data to GPU buffer
    device.queue.writeBuffer(gpuBuffer, 0, inputData);

    // 7. Create bind group (connects buffer to shader binding)
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: gpuBuffer },
        }],
    });

    // 8. Create readback buffer (for copying results back to CPU)
    const readbackBuffer = device.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // 9. Encode and submit compute commands
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    // Dispatch enough workgroups to cover all N elements
    // ceil(N / 64) workgroups
    pass.dispatchWorkgroups(Math.ceil(N / 64));
    pass.end();

    // Copy results to readback buffer
    commandEncoder.copyBufferToBuffer(gpuBuffer, 0,
                                       readbackBuffer, 0,
                                       inputData.byteLength);

    device.queue.submit([commandEncoder.finish()]);

    // 10. Read results back to CPU
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackBuffer.getMappedRange());
    console.log('First 10 results:', Array.from(result.slice(0, 10)));
    // Expected: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    readbackBuffer.unmap();
}

doubleBuffer();
""".strip()

    print("  WGSL Compute Shader:")
    print("  " + "-" * 60)
    for line in wgsl_shader.split('\n'):
        print(f"  {line}")
    print("  " + "-" * 60)
    print()

    print("  JavaScript Host Code:")
    print("  " + "-" * 60)
    for line in js_code.split('\n'):
        print(f"  {line}")
    print("  " + "-" * 60)
    print()

    # --- Python simulation of the compute shader logic ---
    print("  Python Simulation of the Compute Shader:")
    N = 1024
    WORKGROUP_SIZE = 64
    data = np.arange(1, N + 1, dtype=np.float32)  # [1, 2, ..., 1024]

    num_workgroups = int(np.ceil(N / WORKGROUP_SIZE))
    print(f"    Buffer size: {N} elements ({N * 4} bytes)")
    print(f"    Workgroup size: {WORKGROUP_SIZE}")
    print(f"    Workgroups dispatched: {num_workgroups}")

    # Simulate the compute dispatch
    result = data.copy()
    for wg in range(num_workgroups):
        for local_id in range(WORKGROUP_SIZE):
            global_id = wg * WORKGROUP_SIZE + local_id
            if global_id < N:
                result[global_id] = result[global_id] * 2.0

    print(f"    Input  (first 10): {data[:10].tolist()}")
    print(f"    Output (first 10): {result[:10].tolist()}")
    print(f"    Input  (last 5):   {data[-5:].tolist()}")
    print(f"    Output (last 5):   {result[-5:].tolist()}")

    # Verify
    expected = data * 2.0
    assert np.allclose(result, expected), "Simulation mismatch!"
    print("    Verification: PASSED (all elements doubled correctly)")
    print()

    # API call count
    print("  WebGPU API calls for this compute task:")
    api_calls = [
        "navigator.gpu.requestAdapter()",
        "adapter.requestDevice()",
        "device.createShaderModule()",
        "device.createComputePipeline()",
        "device.createBuffer() x2 (storage + readback)",
        "device.queue.writeBuffer()",
        "device.createBindGroup()",
        "device.createCommandEncoder()",
        "encoder.beginComputePass()",
        "pass.setPipeline()",
        "pass.setBindGroup()",
        "pass.dispatchWorkgroups()",
        "pass.end()",
        "encoder.copyBufferToBuffer()",
        "encoder.finish()",
        "device.queue.submit()",
        "readbackBuffer.mapAsync()",
        "readbackBuffer.getMappedRange()",
        "readbackBuffer.unmap()",
    ]
    print(f"    Total: {len(api_calls)} calls")
    for i, call in enumerate(api_calls, 1):
        print(f"      {i:2d}. {call}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Lesson 16: Modern Graphics APIs Overview -- Exercises")
    print("=" * 70)
    print()

    exercise_1()
    print()
    exercise_2()
    print()
    exercise_3()
    print()
    exercise_4()
    print()
    exercise_5()
    print()
    exercise_6()

    print()
    print("=" * 70)
    print("All exercises completed.")
    print("=" * 70)

export type AnygpuBindGroupLayoutEntry = GPUBindGroupLayoutEntry & {name: string};

export type AnygpuBindGroupLayout = GPUBindGroupLayout;

export type AnygpuPipelineLayout = GPUPipelineLayout;

export const AnygpuShaderStage = {
  VERTEX: 1,
  FRAGMENT: 2,
  COMPUTE: 4,
} as const;

export interface AnygpuStageState {
  label?: string;
  vertex?: { buffers: GPUVertexBufferLayout[]; constants?: Record<string, number> };
  fragment?: { targets: GPUColorTargetState[]; constants?: Record<string, number> };
  compute?: { constants?: Record<string, number> };
  primitive?: GPUPrimitiveState;
  depthStencil?: GPUDepthStencilState;
  multisample?: GPUMultisampleState;
}

export interface AnygpuStage {
  entryPoint: string;
  constants: Record<string, number>;
  zeroInitializeWorkgroupMemory?: boolean;
}


export interface AnygpuPipeline {
  bindGroupLayouts: AnygpuBindGroupLayoutEntry[][];
  pushConstantRanges: { stages: GPUShaderStageFlags; start: number; end: number }[];
  vertex?: AnygpuStage;
  fragment?: AnygpuStage;
  compute?: AnygpuStage;
}

export class PipelineHelper<Resouces extends Record<string ,GPUBindingResource> = Record<string, GPUBindingResource>> {
  public constructor(private readonly pipeline: AnygpuPipeline) {}

  private bindGroupLayouts(device: GPUDevice): AnygpuBindGroupLayout[] {
    return this.pipeline.bindGroupLayouts.map((entries, group) =>
      device.createBindGroupLayout({ label: `anygpu group ${group}`, entries }),
    );
  }
  private pipelineLayout(device: GPUDevice): AnygpuPipelineLayout {
    return device.createPipelineLayout({
      bindGroupLayouts: this.bindGroupLayouts(device),
    });
  }

  private static constants(
    stage: AnygpuStage,
    overrides: Record<string, number> | undefined,
  ): Record<string, number> {
    return { ...stage.constants, ...overrides };
  }

  public generateBindGroups(
    device: GPUDevice,
    resources: Resouces,
    layouts?: AnygpuBindGroupLayout[],
  ): AnygpuBindGroupLayout extends GPUBindGroupLayout ? GPUBindGroup[] : never {
    const bgls = layouts ?? this.bindGroupLayouts(device);
    return this.pipeline.bindGroupLayouts.map((entries, group) =>
      device.createBindGroup({
        label: `anygpu group ${group}`,
        layout: bgls[group],
        entries: entries.map((entry) => {
          const resource = resources[entry.name];
          if (resource === undefined) {
            throw new Error(`missing resource for binding "${entry.name}" (group ${group}, binding ${entry.binding})`);
          }
          return { binding: entry.binding, resource };
        }),
      }),
    );
  }

  public generateDescriptor(
    device: GPUDevice,
    shaderModule: GPUShaderModule,
    host: AnygpuStageState = {},
  ): GPURenderPipelineDescriptor | GPUComputePipelineDescriptor {
    const layout = this.pipelineLayout(device);
    const compute = this.pipeline.compute;
    if (compute !== undefined) {
      const state: GPUComputeState & { zeroInitializeWorkgroupMemory?: boolean } = {
        module: shaderModule,
        entryPoint: compute.entryPoint,
        constants: {...compute.constants, ...host.compute?.constants}
      };
      if (compute.zeroInitializeWorkgroupMemory === true) {
        state.zeroInitializeWorkgroupMemory = true;
      }
      const descriptor: GPUComputePipelineDescriptor = { layout, compute: state };
      if (host.label !== undefined) { descriptor.label = host.label; }
      return descriptor;
    }

    const vertex = this.pipeline.vertex;
    if (vertex === undefined) {
      throw new Error(
        `${this.constructor.name} has no vertex stage, so no render pipeline can be made from it`,
      );
    }

    const descriptor: GPURenderPipelineDescriptor = {
      layout,
      vertex: {
        module: shaderModule,
        entryPoint: vertex.entryPoint,
        buffers: host.vertex?.buffers ?? [],
        constants: PipelineHelper.constants(vertex, host.vertex?.constants),
      },
    };

    const fragment = this.pipeline.fragment;
    if (fragment !== undefined) {
      const targets = host.fragment?.targets;
      if (targets === undefined) {
        throw new Error(
          `${this.constructor.name} has a fragment stage, so fragment.targets has to come from the host`,
        );
      }
      descriptor.fragment = {
        module: shaderModule,
        entryPoint: fragment.entryPoint,
        targets,
        constants: PipelineHelper.constants(fragment, host.fragment?.constants),
      };
    }

    if (host.label !== undefined) { descriptor.label = host.label; }
    if (host.primitive !== undefined) { descriptor.primitive = host.primitive; }
    if (host.depthStencil !== undefined) { descriptor.depthStencil = host.depthStencil; }
    if (host.multisample !== undefined) { descriptor.multisample = host.multisample; }
    return descriptor;
  }
}

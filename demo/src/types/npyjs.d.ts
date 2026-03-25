declare module "npyjs" {
  export interface NpyArray {
    data: Float32Array;
    shape: number[];
  }

  export default class Npyjs {
    load(path: string): Promise<NpyArray>;
  }
}

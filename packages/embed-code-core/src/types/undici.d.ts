// Type declarations for modules without @types packages

declare module 'undici' {
  export class ProxyAgent {
    constructor(options: {
      uri: string;
      keepAliveTimeout?: number;
      keepAliveMaxTimeout?: number;
    });
  }
}

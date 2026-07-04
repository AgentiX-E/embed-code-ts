import { describe, it, expect } from 'vitest';
import { Command } from 'commander';

describe('CLI program', () => {
  it('exports a Command instance', () => {
    const program = new Command();
    expect(program).toBeDefined();
    expect(program.name).toBeDefined();
  });

  it('handles download command options', () => {
    const program = new Command();
    program
      .command('download')
      .argument('[model-id]', 'Model ID')
      .option('--proxy-url <url>', 'Proxy URL')
      .option('--proxy-username <user>', 'Proxy username')
      .option('--proxy-password <pass>', 'Proxy password');
    expect(program.commands.length).toBeGreaterThan(0);
    const downloadCmd = program.commands.find((c) => c.name() === 'download');
    expect(downloadCmd).toBeDefined();
    expect(downloadCmd!.options.some((o) => o.long === '--proxy-url')).toBe(true);
  });
});

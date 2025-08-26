/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {Client as McpClient} from '@modelcontextprotocol/sdk/client/index.js';
import {InMemoryTransport} from '@modelcontextprotocol/sdk/inMemory.js';
import {McpServer} from '@modelcontextprotocol/sdk/server/mcp.js';
import {z} from 'zod';

import {mcpToTool} from '../../../src/mcp/_mcp.js';
import {GoogleGenAI} from '../../../src/node/node_client.js';
import {
  FunctionCallingConfigMode,
  FunctionDeclaration,
  HttpOptions,
  Type,
} from '../../../src/types.js';
import {setupTestServer, shutdownTestServer} from '../test_server.js';

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const GOOGLE_CLOUD_PROJECT = process.env.GOOGLE_CLOUD_PROJECT;
const GOOGLE_CLOUD_LOCATION = process.env.GOOGLE_CLOUD_LOCATION;

const controlLightFunctionDeclaration: FunctionDeclaration = {
  name: 'controlLight',
  parameters: {
    type: Type.OBJECT,
    description: 'Set the brightness and color temperature of a room light.',
    properties: {
      brightness: {
        type: Type.NUMBER,
        description:
          'Light level from 0 to 100. Zero is off and 100 is full brightness.',
      },
      colorTemperature: {
        type: Type.STRING,
        description:
          'Color temperature of the light fixture which can be `daylight`, `cool` or `warm`.',
      },
    },
    required: ['brightness', 'colorTemperature'],
  },
};

describe('MCP related client Tests', () => {
  let testName: string = '';
  let httpOptions: HttpOptions;
  beforeAll(async () => {
    await setupTestServer();
    jasmine.getEnv().addReporter({
      specStarted: function (result) {
        testName = result.fullName;
      },
    });
  });

  afterAll(async () => {
    await shutdownTestServer();
  });

  beforeEach(() => {
    httpOptions = {headers: {'Test-Name': testName}};
  });

  describe('generateContent', () => {
    it('ML Dev one CallableTool with MCPClients and conduct automated function calling', async () => {
      const ai = new GoogleGenAI({
        vertexai: false,
        apiKey: GOOGLE_API_KEY,
        httpOptions,
      });
      const mcpCallableTool = mcpToTool(
        await spinUpPrintingServer(),
        await spinUpBeepingServer(),
      );
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents:
          'Use the printer to print a simple word: hello in blue, and beep with the beeper, remember to print and beep',
        config: {
          tools: [mcpCallableTool],
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
            },
          },
        },
      });
      const expectedFunctionResponse = {
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: 'print',
              response: {
                content: [
                  {
                    type: 'text',
                    text: '\u001b[34mhello\u001b[0m',
                  },
                ],
              },
            },
          },
          {
            functionResponse: {
              name: 'beep',
              response: {
                content: [
                  {
                    type: 'text',
                    text: 'beep',
                  },
                ],
              },
            },
          },
        ],
      };
      expect(response.automaticFunctionCallingHistory![2]).toEqual(
        expectedFunctionResponse,
      );
    });
    it('ML Dev test with greeter server (parameter as nullable union type)', async () => {
      const ai = new GoogleGenAI({
        vertexai: false,
        apiKey: GOOGLE_API_KEY,
        httpOptions,
      });
      const mcpCallableTool = mcpToTool(await greetServer());
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents:
          'call the greeter once with name: jone smith, and greeting: Hello',
        config: {
          tools: [mcpCallableTool],
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
            },
          },
          automaticFunctionCalling: {
            maximumRemoteCalls: 1,
          },
        },
      });
      const expectedFunctionResponse = {
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: 'greet',
              response: {
                content: [
                  {
                    type: 'text',
                    text: 'User give name as a string: Hello jone smith: Hello',
                  },
                ],
              },
            },
          },
        ],
      };
      expect(response.automaticFunctionCallingHistory![2]).toEqual(
        expectedFunctionResponse,
      );
    });
    it('Vertex AI test with greeter server (parameter as nullable union type)', async () => {
      const ai = new GoogleGenAI({
        vertexai: true,
        project: GOOGLE_CLOUD_PROJECT,
        location: GOOGLE_CLOUD_LOCATION,
        httpOptions,
      });
      const mcpCallableTool = mcpToTool(await greetServer());
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents:
          'call the greeter once with name: jone smith, and greeting: Hello',
        config: {
          tools: [mcpCallableTool],
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
            },
          },
          automaticFunctionCalling: {
            maximumRemoteCalls: 1,
          },
        },
      });
      const expectedFunctionResponse = {
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: 'greet',
              response: {
                content: [
                  {
                    type: 'text',
                    text: 'User give name as a string: Hello jone smith: Hello',
                  },
                ],
              },
            },
          },
        ],
      };
      expect(response.automaticFunctionCallingHistory![2]).toEqual(
        expectedFunctionResponse,
      );
    });
    it('ML Dev Multiple CallableTool with MCPClients and conduct automated function calling', async () => {
      const ai = new GoogleGenAI({
        vertexai: false,
        apiKey: GOOGLE_API_KEY,
        httpOptions,
      });
      const callableTool1 = mcpToTool(await spinUpPrintingServer());
      const callableTool2 = mcpToTool(await spinUpBeepingServer());
      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents:
          'Use the printer to print a simple word: hello in blue, and beep with the beeper, make sure you beep',
        config: {
          tools: [callableTool1, callableTool2],
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
            },
          },
        },
      });
      const expectedFunctionResponse = {
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: 'print',
              response: {
                content: [
                  {
                    type: 'text',
                    text: '\u001b[34mhello\u001b[0m',
                  },
                ],
              },
            },
          },
          {
            functionResponse: {
              name: 'beep',
              response: {
                content: [
                  {
                    type: 'text',
                    text: 'beep',
                  },
                ],
              },
            },
          },
        ],
      };
      expect(response.automaticFunctionCallingHistory![2]).toEqual(
        expectedFunctionResponse,
      );
    });
    it('Vertex AI should take a list of MCPClients and conduct automated function calling', async () => {
      const ai = new GoogleGenAI({
        vertexai: true,
        project: GOOGLE_CLOUD_PROJECT,
        location: GOOGLE_CLOUD_LOCATION,
        httpOptions,
      });
      const mcpCallableTool = mcpToTool(await spinUpPrintingServer());
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: 'Use the printer to print a simple word: hello in red',
        config: {
          tools: [mcpCallableTool],
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
              // allowedFunctionNames: ['print', 'beep'],
            },
          },
        },
      });
      const expectedFunctionResponse = {
        role: 'user',
        parts: [
          {
            functionResponse: {
              name: 'print',
              response: {
                content: [
                  {
                    type: 'text',
                    text: '\u001b[31mhello\u001b[0m',
                  },
                ],
              },
            },
          },
        ],
      };
      expect(response.automaticFunctionCallingHistory![2]).toEqual(
        expectedFunctionResponse,
      );
    });
    it('ML Dev will give FunctionDeclaration back when AFC is disabled', async () => {
      const ai = new GoogleGenAI({
        vertexai: false,
        apiKey: GOOGLE_API_KEY,
        httpOptions,
      });
      const callableTool1 = mcpToTool(await spinUpPrintingServer());
      const callableTool2 = mcpToTool(await spinUpBeepingServer());
      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents:
          'Use the printer to print a simple word: hello in blue, and beep with the beeper',
        config: {
          tools: [callableTool1, callableTool2],
          automaticFunctionCalling: {
            disable: true,
          },
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
            },
          },
        },
      });
      const expectedFunctionCalls = [
        {
          name: 'print',
          args: {
            text: 'hello',
            color: 'blue',
          },
        },
        {
          name: 'beep',
          args: {},
        },
      ];
      expect(response.functionCalls).toEqual(expectedFunctionCalls);
    });
    it('Vertex AI will give FunctionDeclaration back when AFC is disabled', async () => {
      const ai = new GoogleGenAI({
        vertexai: true,
        project: GOOGLE_CLOUD_PROJECT,
        location: GOOGLE_CLOUD_LOCATION,
        httpOptions,
      });
      const mcpCallableTool = mcpToTool(await spinUpPrintingServer());
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: 'Use the printer to print a simple word: hello in blue',
        config: {
          tools: [mcpCallableTool],
          automaticFunctionCalling: {
            disable: true,
          },
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
            },
          },
        },
      });
      const expectedFunctionCalls = [
        {
          name: 'print',
          args: {
            text: 'hello',
            color: 'blue',
          },
        },
      ];
      expect(response.functionCalls).toEqual(expectedFunctionCalls);
    });
    it('ML Dev can take mixed tools when AFC is disabled', async () => {
      const ai = new GoogleGenAI({
        vertexai: false,
        apiKey: GOOGLE_API_KEY,
        httpOptions,
      });
      const callableTool1 = mcpToTool(await spinUpPrintingServer());
      const callableTool2 = mcpToTool(await spinUpBeepingServer());
      const consoleLogSpy = spyOn(console, 'log').and.callThrough();
      const consoleBeepSpy = spyOn(process.stdout, 'write').and.callThrough();
      const response = await ai.models.generateContent({
        model: 'gemini-2.0-flash',
        contents:
          'Use the printer to print a simple word: hello in blue, and beep with the beeper, then control the light to warm, 50',
        config: {
          tools: [
            callableTool1,
            callableTool2,
            {functionDeclarations: [controlLightFunctionDeclaration]},
          ],
          automaticFunctionCalling: {
            disable: true,
          },
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
            },
          },
        },
      });
      const expectedFunctionCalls = [
        {
          name: 'print',
          args: {
            text: 'hello',
            color: 'blue',
          },
        },
        {
          name: 'beep',
          args: {},
        },
        {
          name: 'controlLight',
          args: {
            brightness: 50,
            colorTemperature: 'warm',
          },
        },
      ];
      expect(response.functionCalls).toEqual(expectedFunctionCalls);
      expect(consoleLogSpy).not.toHaveBeenCalled();
      expect(consoleBeepSpy).not.toHaveBeenCalledWith('\u0007');
    });
    it('Vertex AI can take function declarations when AFC is disabled', async () => {
      const ai = new GoogleGenAI({
        vertexai: true,
        project: GOOGLE_CLOUD_PROJECT,
        location: GOOGLE_CLOUD_LOCATION,
        httpOptions,
      });
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: 'control the light to warm, 50',
        config: {
          tools: [{functionDeclarations: [controlLightFunctionDeclaration]}],
          automaticFunctionCalling: {
            disable: true,
          },
          toolConfig: {
            functionCallingConfig: {
              mode: FunctionCallingConfigMode.AUTO,
            },
          },
        },
      });
      const expectedFunctionCalls = [
        {
          name: 'controlLight',
          args: {
            brightness: 50,
            colorTemperature: 'warm',
          },
        },
      ];
      expect(response.functionCalls).toEqual(expectedFunctionCalls);
    });
  });
});

async function spinUpPrintingServer(): Promise<McpClient> {
  const server = new McpServer({
    name: 'printer',
    version: '1.0.0',
  });

  const colorMap: {[key: string]: string} = {
    red: '\x1b[31m',
    green: '\x1b[32m',
    blue: '\x1b[34m',
    white: '\x1b[37m',
    reset: '\x1b[0m', // Resets all text attributes to default
  };

  server.tool(
    'print',
    {
      text: z.string(),
      color: z.string().regex(/red|blue|green|white/),
    },
    async ({text, color}) => {
      if (colorMap[color]) {
        console.log(colorMap[color] + text);
        console.log(colorMap.reset);
      } else {
        console.log(text);
      }

      return {
        content: [
          {
            type: 'text',
            text: colorMap[color] + text + colorMap.reset,
          },
        ],
      };
    },
  );

  const transports = InMemoryTransport.createLinkedPair();
  await server.connect(transports[0]);

  const client = new McpClient({
    name: 'printer',
    version: '1.0.0',
  });
  client.connect(transports[1]);

  return client;
}

async function spinUpBeepingServer(): Promise<McpClient> {
  const server = new McpServer({
    name: 'beeper',
    version: '1.0.0',
  });

  server.tool('beep', async () => {
    process.stdout.write('\u0007');
    return {
      content: [
        {
          type: 'text',
          text: 'beep',
        },
      ],
    };
  });

  const transports = InMemoryTransport.createLinkedPair();
  await server.connect(transports[0]);

  const client = new McpClient({
    name: 'beeper',
    version: '1.0.0',
  });
  client.connect(transports[1]);

  return client;
}

async function greetServer(): Promise<McpClient> {
  const server = new McpServer({
    name: 'greeter',
    version: '1.0.0',
  });

  server.tool(
    'greet',
    {
      name: z.union([z.string(), z.number(), z.null()]),
      greeting: z.string(),
    },
    async ({name, greeting}) => {
      let greetMessage = '';
      if (typeof name === 'string') {
        greetMessage = `User give name as a string: Hello ${name}: ${greeting}`;
      } else {
        greetMessage = `User give name as a number: Hello ${name}: ${greeting}`;
      }
      return {
        content: [
          {
            type: 'text',
            text: greetMessage,
          },
        ],
      };
    },
  );

  const transports = InMemoryTransport.createLinkedPair();
  await server.connect(transports[0]);

  const client = new McpClient({
    name: 'greeter',
    version: '1.0.0',
  });
  client.connect(transports[1]);

  return client;
}

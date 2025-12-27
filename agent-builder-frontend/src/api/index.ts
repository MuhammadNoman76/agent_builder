// src/api/index.ts
import axios, { AxiosInstance, AxiosError } from 'axios';
import { 
  User, 
  Flow, 
  AuthTokens, 
  ComponentConfig,
  ApiError 
} from '../types';

const API_BASE_URL = '/api/v1';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for auth token
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ApiError>) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('access_token');
          window.location.href = '/auth';
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth endpoints
  async register(username: string, email: string, password: string): Promise<User> {
    const response = await this.client.post('/auth/register', {
      username,
      email,
      password,
    });
    return response.data;
  }

  async login(email: string, password: string): Promise<AuthTokens> {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);

    const response = await this.client.post('/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get('/auth/me');
    return response.data;
  }

  // Flow endpoints
  async getFlows(page = 1, pageSize = 10): Promise<{
    flows: Flow[];
    total: number;
    page: number;
    page_size: number;
  }> {
    const response = await this.client.get('/flows', {
      params: { page, page_size: pageSize },
    });
    return response.data;
  }

  async getFlow(flowId: string): Promise<Flow> {
    const response = await this.client.get(`/flows/${flowId}`);
    return response.data;
  }

  async createFlow(data: {
    name: string;
    description?: string;
    nodes?: any[];
    edges?: any[];
  }): Promise<Flow> {
    const response = await this.client.post('/flows', data);
    return response.data;
  }

  async updateFlow(flowId: string, data: Partial<{
    name: string;
    description: string;
    nodes: any[];
    edges: any[];
    is_active: boolean;
  }>): Promise<Flow> {
    const response = await this.client.put(`/flows/${flowId}`, data);
    return response.data;
  }

  async deleteFlow(flowId: string): Promise<void> {
    await this.client.delete(`/flows/${flowId}`);
  }

  async executeFlow(flowId: string, inputData: Record<string, any>): Promise<{
    flow_id: string;
    status: string;
    output?: any;
    error?: string;
    execution_time?: number;
  }> {
    const response = await this.client.post(`/flows/${flowId}/execute`, {
      input_data: inputData,
    });
    return response.data;
  }

  async getFlowSchema(flowId: string): Promise<{
    flow_id: string;
    agent_schema: any;
    validation_errors: string[];
    is_valid: boolean;
  }> {
    const response = await this.client.get(`/flows/${flowId}/schema`);
    return response.data;
  }

  // Export flow as JSON
  async exportFlow(flowId: string): Promise<{
    flow: Flow;
    export_date: string;
    version: string;
  }> {
    const response = await this.client.get(`/flows/${flowId}/export`);
    return response.data;
  }

  // Import flow from JSON
  async importFlow(flowData: any): Promise<Flow> {
    const response = await this.client.post('/flows/import', flowData);
    return response.data;
  }

  // Component endpoints
  async getComponents(category?: string): Promise<{
    components: Array<{
      component_type: string;
      config: ComponentConfig;
    }>;
    total: number;
  }> {
    const response = await this.client.get('/components', {
      params: category ? { category } : {},
    });
    return response.data;
  }

  async getComponentCategories(): Promise<{
    categories: string[];
    components_by_category: Record<string, Array<{
      component_type: string;
      config: ComponentConfig;
    }>>;
  }> {
    const response = await this.client.get('/components/categories');
    return response.data;
  }

  async getComponent(componentType: string): Promise<{
    component_type: string;
    config: ComponentConfig;
  }> {
    const response = await this.client.get(`/components/${componentType}`);
    return response.data;
  }

  async getComponentSchema(componentType: string): Promise<any> {
    const response = await this.client.get(`/components/${componentType}/schema`);
    return response.data;
  }

  async validateEdge(data: {
    source_component_type: string;
    source_port: string;
    target_component_type: string;
    target_port: string;
  }): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    const response = await this.client.post('/edges/validate', data);
    return response.data;
  }

  // Node operations
  async addNode(flowId: string, nodeData: any): Promise<Flow> {
    const response = await this.client.post(`/flows/${flowId}/nodes`, nodeData);
    return response.data;
  }

  async updateNode(flowId: string, nodeId: string, updateData: any): Promise<Flow> {
    const response = await this.client.put(`/flows/${flowId}/nodes/${nodeId}`, updateData);
    return response.data;
  }

  async deleteNode(flowId: string, nodeId: string): Promise<Flow> {
    const response = await this.client.delete(`/flows/${flowId}/nodes/${nodeId}`);
    return response.data;
  }

  // Edge operations
  async addEdge(flowId: string, edgeData: any): Promise<Flow> {
    const response = await this.client.post(`/flows/${flowId}/edges`, edgeData);
    return response.data;
  }

  async deleteEdge(flowId: string, edgeId: string): Promise<Flow> {
    const response = await this.client.delete(`/flows/${flowId}/edges/${edgeId}`);
    return response.data;
  }
}

export const api = new ApiClient();
export default api;

import axios from 'axios';
import { Corpus, QueryResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const ragService = {
  async listCorpora(): Promise<Corpus[]> {
    const response = await api.get('/corpora');
    return response.data;
  },

  async createCorpus(name: string, description?: string): Promise<Corpus> {
    const response = await api.post('/corpora', {
      name,
      description,
    });
    return response.data;
  },

  async deleteCorpus(corpusId: string): Promise<void> {
    await api.delete(`/corpora/${corpusId}`);
  },

  async uploadDocument(corpusId: string, file: File): Promise<void> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('corpus_id', corpusId);

    await api.post('/documents', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  async deleteDocument(documentId: string): Promise<void> {
    await api.delete(`/documents/${documentId}`);
  },

  async query(corpusId: string, query: string, numResults: number = 5): Promise<QueryResponse> {
    const response = await api.post('/query', {
      corpus_id: corpusId,
      query,
      num_results: numResults,
    });
    return response.data;
  },

  async getCorpusInfo(corpusId: string): Promise<Corpus> {
    const response = await api.get(`/corpora/${corpusId}`);
    return response.data;
  },
};
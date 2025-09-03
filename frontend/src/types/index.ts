export interface Corpus {
  id: string;
  name: string;
  description?: string;
  document_count: number;
  created_at?: string;
  updated_at?: string;
}

export interface Document {
  id: string;
  corpus_id: string;
  title: string;
  content: string;
  metadata?: Record<string, any>;
  created_at?: string;
}

export interface QueryRequest {
  corpus_id: string;
  query: string;
  num_results?: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  confidence: number;
}

export interface Source {
  text: string;
  metadata: Record<string, any>;
  score: number;
}
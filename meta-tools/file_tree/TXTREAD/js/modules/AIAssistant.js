// AIAssistant.js - AI chat functionality with Gemini API
export class AIAssistant {
    constructor() {
        this.apiKey = '';
        this.currentText = '';
        this.messages = [];
        this.maxContextLength = 8000;
        this.debugMode = false;
    }

    setApiKey(key) {
        this.apiKey = key;
        return !!this.apiKey;
    }

    setCurrentText(text) {
        this.currentText = text;
    }

    addMessage(type, content) {
        const message = {
            type,
            content,
            timestamp: new Date()
        };
        this.messages.push(message);
        return message;
    }

    async sendMessage(userMessage) {
        if (!this.apiKey) {
            throw new Error('API key is required');
        }

        if (!this.currentText) {
            throw new Error('No text loaded for analysis');
        }

        this.addMessage('user', userMessage);

        try {
            const response = await this.callGeminiAPI(userMessage);
            this.addMessage('ai', response);
            return response;
        } catch (error) {
            console.error('AI Assistant error:', error);
            throw error;
        }
    }

    async callGeminiAPI(userMessage) {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${this.apiKey}`;
        
        const selectedText = this.getSelectedText();
        const contextText = selectedText || this.currentText;
        
        // Limit context to avoid token limits
        const truncatedContext = contextText.length > this.maxContextLength 
            ? contextText.substring(0, this.maxContextLength) + '...'
            : contextText;
        
        const prompt = this.buildPrompt(userMessage, truncatedContext);

        const requestBody = {
            contents: [{
                parts: [{
                    text: prompt
                }]
            }],
            generationConfig: {
                temperature: 0.7,
                topK: 40,
                topP: 0.95,
                maxOutputTokens: 2048
            },
            safetySettings: [
                {
                    category: "HARM_CATEGORY_HATE_SPEECH",
                    threshold: "BLOCK_ONLY_HIGH"
                },
                {
                    category: "HARM_CATEGORY_DANGEROUS_CONTENT", 
                    threshold: "BLOCK_ONLY_HIGH"
                },
                {
                    category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold: "BLOCK_ONLY_HIGH"
                },
                {
                    category: "HARM_CATEGORY_HARASSMENT",
                    threshold: "BLOCK_ONLY_HIGH"
                }
            ]
        };

        if (this.debugMode) {
            console.log('Gemini API Request:', {
                url: url.replace(this.apiKey, 'API_KEY_HIDDEN'),
                promptLength: prompt.length,
                promptPreview: prompt.substring(0, 100) + '...'
            });
        }

        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.text();
            let errorMessage = `API request failed: ${response.status}`;
            
            try {
                const errorJson = JSON.parse(errorData);
                if (errorJson.error && errorJson.error.message) {
                    errorMessage = this.parseApiError(errorJson.error.message, response.status);
                }
            } catch (e) {
                errorMessage = this.getGenericErrorMessage(response.status);
            }
            
            throw new Error(errorMessage);
        }

        const data = await response.json();
        
        if (!data || !data.candidates || !Array.isArray(data.candidates) || data.candidates.length === 0) {
            throw new Error('Invalid API response: No candidates found.');
        }
        
        const candidate = data.candidates[0];
        
        if (candidate.finishReason === 'SAFETY') {
            throw new Error('The response was blocked due to safety filters. Try rephrasing your question.');
        }
        
        if (!candidate.content || !candidate.content.parts || !Array.isArray(candidate.content.parts) || 
            candidate.content.parts.length === 0 || !candidate.content.parts[0].text) {
            throw new Error('Invalid API response: No content found.');
        }
        
        return candidate.content.parts[0].text.trim();
    }

    buildPrompt(userMessage, context) {
        const lowerMessage = userMessage.toLowerCase();
        
        if (lowerMessage.includes('quiz')) {
            return `You are an expert educator. Based on the following text, create 5 thoughtful multiple-choice questions that test deep comprehension and critical thinking. Include questions about main ideas, supporting details, inferences, and applications. Format each question with the question, 4 options (A, B, C, D), and indicate the correct answer with an explanation.

Text: ${context}

Create questions that require understanding, not just memorization.`;
        } else if (lowerMessage.includes('summarize')) {
            return `You are an expert reader and analyst. Provide a comprehensive summary of the following text. Include:
- Main thesis or central idea
- Key supporting points (3-5 bullet points)
- Important details or examples
- Conclusion or implications

Text: ${context}

Be thorough but concise.`;
        } else if (lowerMessage.includes('key points')) {
            return `You are an expert at identifying important information. Extract and explain the key points from the following text. For each key point:
- State the main idea clearly
- Explain why it's important
- Note any supporting evidence

Text: ${context}

Focus on the most significant ideas.`;
        } else if (lowerMessage.includes('explain')) {
            return `You are an expert teacher who excels at making complex topics simple. Analyze the following text and:
- Identify all difficult concepts, technical terms, or jargon
- Explain each in simple, everyday language
- Provide examples or analogies where helpful
- Clarify any ambiguous passages

Text: ${context}

Make everything crystal clear for a general audience.`;
        } else if (lowerMessage.includes('analyze')) {
            return `You are a critical thinking expert. Provide a detailed analysis of the following text including:
- Author's purpose and intended audience
- Main arguments or claims
- Evidence and reasoning used
- Strengths and potential weaknesses
- Overall effectiveness

Text: ${context}

Be thorough and objective in your analysis.`;
        } else {
            return `You are an expert reading assistant with deep knowledge across all subjects. You've carefully studied the following text and understand it completely:

---
${context}
---

User question: ${userMessage}

Provide a helpful, accurate, and detailed response. If the question asks about specific details from the text, quote relevant passages. If it requires analysis or interpretation, provide thoughtful insights. Always base your answer on the actual content of the text.`;
        }
    }

    parseApiError(errorMessage, statusCode) {
        if (errorMessage.includes('API_KEY_INVALID')) {
            return 'Invalid API key. Please check your Gemini API key.';
        } else if (errorMessage.includes('QUOTA_EXCEEDED')) {
            return 'API quota exceeded. Please check your Google Cloud billing.';
        } else if (errorMessage.includes('RATE_LIMIT_EXCEEDED')) {
            return 'Rate limit exceeded. Please wait a moment and try again.';
        }
        
        return this.getGenericErrorMessage(statusCode);
    }

    getGenericErrorMessage(statusCode) {
        switch (statusCode) {
            case 400:
                return 'Bad request. The API key may be invalid or the request format is incorrect.';
            case 401:
                return 'Authentication failed. Please check your API key.';
            case 403:
                return 'Access forbidden. Your API key may not have the necessary permissions.';
            case 429:
                return 'Too many requests. Please wait a moment and try again.';
            default:
                return `API request failed with status ${statusCode}`;
        }
    }

    getSelectedText() {
        // This should be connected to the actual text selection in the UI
        if (window.getSelection) {
            return window.getSelection().toString();
        }
        return '';
    }

    clearMessages() {
        this.messages = [];
    }

    getMessages() {
        return this.messages;
    }

    setDebugMode(enabled) {
        this.debugMode = enabled;
    }

    // Quick action methods
    async generateQuiz() {
        return this.sendMessage('Create a quiz based on this text');
    }

    async generateSummary() {
        return this.sendMessage('Summarize this text');
    }

    async extractKeyPoints() {
        return this.sendMessage('What are the key points in this text?');
    }

    async explainConcepts() {
        return this.sendMessage('Explain the difficult concepts in this text');
    }

    async analyzeText() {
        return this.sendMessage('Provide a detailed analysis of this text');
    }
}
'use client';

import { useState } from 'react';

// 추출된 파라미터 타입 정의
interface ExtractedParams {
  date?: string;
  day_type?: string;
  weather?: string;
  temperature?: number;
  precipitation?: number;
  time_slots?: string[];
  categories?: string[];
  products?: string[];
}

// 메시지 타입을 정의합니다.
interface Message {
  role: 'user' | 'assistant';
  content: string;
  recommendations?: Recommendation[];
  extractedParams?: ExtractedParams;
}

// 추천 결과 타입 정의
interface Recommendation {
  time_slot: string;
  features: {
    product_name: string;
    // 필요시 추가 필드 작성
  };
  predicted_sales: number;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    const newMessages: Message[] = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/v1/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_query: input }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      // 백엔드 응답을 기반으로 어시스턴트 메시지를 생성합니다.
      const recommendations = data.recommendations;
      const extractedParams = data.extracted_params;
      let assistantMessageContent;

      if (recommendations && recommendations.length > 0) {
        assistantMessageContent = "다음 편성을 추천합니다:";
      } else {
        assistantMessageContent = "추천할 만한 방송을 찾지 못했습니다.";
      }

      const assistantMessage: Message = { 
        role: 'assistant', 
        content: assistantMessageContent,
        recommendations: recommendations && recommendations.length > 0 ? recommendations : undefined,
        extractedParams: extractedParams
      };

      setMessages(prevMessages => [...prevMessages, assistantMessage]);

    } catch (error) {
      console.error('Error fetching recommendation:', error);
      const errorMessage: Message = { role: 'assistant', content: '죄송합니다. 추천을 받아오는 중 오류가 발생했습니다.' };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col h-[70vh] border rounded-lg shadow-lg">
      <div className="flex-1 p-4 overflow-y-auto">
        {messages.map((msg, index) => (
          <div key={index} className={`my-2 p-2 rounded-lg ${msg.role === 'user' ? 'bg-blue-100 text-right ml-auto' : 'bg-gray-100 text-left mr-auto'}`}>
            <div className="whitespace-pre-wrap">{msg.content}</div>
            {msg.extractedParams && (
              <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                <h4 className="font-semibold text-blue-800 mb-2">📊 분석된 파라미터</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  {msg.extractedParams.date && (
                    <div><span className="font-medium text-gray-600">날짜:</span> {msg.extractedParams.date}</div>
                  )}
                  {msg.extractedParams.day_type && (
                    <div><span className="font-medium text-gray-600">요일:</span> {msg.extractedParams.day_type}</div>
                  )}
                  {msg.extractedParams.weather && (
                    <div><span className="font-medium text-gray-600">날씨:</span> {msg.extractedParams.weather}</div>
                  )}
                  {msg.extractedParams.temperature && (
                    <div><span className="font-medium text-gray-600">온도:</span> {msg.extractedParams.temperature}°C</div>
                  )}
                  {msg.extractedParams.precipitation !== undefined && (
                    <div><span className="font-medium text-gray-600">강수량:</span> {msg.extractedParams.precipitation}mm</div>
                  )}
                  {msg.extractedParams.time_slots && (
                    <div className="col-span-2"><span className="font-medium text-gray-600">시간대:</span> {Array.isArray(msg.extractedParams.time_slots) ? msg.extractedParams.time_slots.join(', ') : msg.extractedParams.time_slots}</div>
                  )}
                  {msg.extractedParams.categories && (
                    <div className="col-span-2"><span className="font-medium text-gray-600">카테고리:</span> {Array.isArray(msg.extractedParams.categories) ? msg.extractedParams.categories.join(', ') : msg.extractedParams.categories}</div>
                  )}
                  {msg.extractedParams.products && (
                    <div className="col-span-2"><span className="font-medium text-gray-600">상품:</span> {Array.isArray(msg.extractedParams.products) ? msg.extractedParams.products.join(', ') : msg.extractedParams.products}</div>
                  )}
                </div>
              </div>
            )}
            {msg.recommendations && (
              <div className="mt-3">
                <table className="w-full border-collapse border border-gray-300 bg-white rounded-lg overflow-hidden">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="py-2 px-4 border-r border-gray-300 text-left font-semibold">시간대</th>
                      <th className="py-2 px-4 border-r border-gray-300 text-left font-semibold">상품명</th>
                      <th className="py-2 px-4 text-right font-semibold">예상 매출</th>
                    </tr>
                  </thead>
                  <tbody>
                    {msg.recommendations.map((rec, recIndex) => (
                      <tr key={recIndex} className="border-b">
                        <td className="py-2 px-4 border-r">{rec.time_slot}</td>
                        <td className="py-2 px-4 border-r">{rec.features.product_name}</td>
                        <td className="py-2 px-4 text-right">{Math.round(rec.predicted_sales).toLocaleString()}원</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="p-4 border-t flex">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-200"
          placeholder="편성 질문을 입력하세요…"
          disabled={isLoading}
        />
        <button
          onClick={handleSendMessage}
          className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-blue-300"
          disabled={isLoading}
        >
          {isLoading ? '전송 중...' : '전송'}
        </button>
      </div>
    </div>
  );
}

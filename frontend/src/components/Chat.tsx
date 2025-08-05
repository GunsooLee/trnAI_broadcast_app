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
  keywords?: string[];
  mode?: string;
}

// 메시지 타입을 정의합니다.
interface Message {
  role: 'user' | 'assistant';
  content: string;
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
  const [isLoadingParams, setIsLoadingParams] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [extractedParams, setExtractedParams] = useState<ExtractedParams | null>(null);
  const [editableParams, setEditableParams] = useState<ExtractedParams | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoadingParams) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoadingParams(true);
    setExtractedParams(null);
    setEditableParams(null);
    setRecommendations([]);

    try {
      const response = await fetch('/api/v1/extract-params', {
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
      const params = data.extracted_params;
      
      setExtractedParams(params);
      setEditableParams({ ...params });

      const assistantMessage: Message = { 
        role: 'assistant', 
        content: "파라미터를 확인하고 필요시 수정한 후 '방송편성 추천' 버튼을 클릭해주세요."
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Error extracting parameters:', error);
      const errorMessage: Message = { role: 'assistant', content: '죄송합니다. 파라미터 추출 중 오류가 발생했습니다.' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoadingParams(false);
    }
  };

  const handleAnalyze = async () => {
    if (!editableParams || isAnalyzing) return;

    setIsAnalyzing(true);
    setRecommendations([]);

    try {
      const response = await fetch('/api/v1/recommend-with-params', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(editableParams),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setRecommendations(data.recommendations || []);

      const resultMessage: Message = { 
        role: 'assistant', 
        content: `분석이 완료되었습니다. ${data.recommendations?.length || 0}개의 추천 결과를 확인해보세요.`
      };
      setMessages(prev => [...prev, resultMessage]);

    } catch (error) {
      console.error('Error getting recommendations:', error);
      const errorMessage: Message = { role: 'assistant', content: '죄송합니다. 추천 분석 중 오류가 발생했습니다.' };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const updateParam = (key: keyof ExtractedParams, value: any) => {
    if (!editableParams) return;
    setEditableParams(prev => prev ? { ...prev, [key]: value } : null);
  };

  return (
    <div className="flex h-[90vh] gap-4">
      {/* 채팅창 */}
      <div className="w-1/2 flex flex-col border rounded-lg shadow-lg">
        <div className="flex-1 p-4 overflow-y-auto">
          {messages.map((msg, index) => (
            <div key={index} className={`my-2 p-2 rounded-lg ${msg.role === 'user' ? 'bg-blue-100 text-right ml-auto' : 'bg-gray-100 text-left mr-auto'}`}>
              <div className="whitespace-pre-wrap">{msg.content}</div>
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
            disabled={isLoadingParams}
          />
          <button
            onClick={handleSendMessage}
            className="ml-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-blue-300"
            disabled={isLoadingParams}
          >
            {isLoadingParams ? '분석 중...' : '전송'}
          </button>
        </div>
      </div>

      {/* 파라미터 패널 */}
      <div className="w-1/4 border rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-bold mb-4">📊 분석 파라미터</h3>
        
        {editableParams ? (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">날짜</label>
              <input
                type="date"
                value={editableParams.date || ''}
                onChange={(e) => updateParam('date', e.target.value)}
                className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">날씨</label>
              <select
                value={editableParams.weather || ''}
                onChange={(e) => updateParam('weather', e.target.value)}
                className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">선택하세요</option>
                <option value="맑음">맑음</option>
                <option value="흐림">흐림</option>
                <option value="비">비</option>
                <option value="눈">눈</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">온도 (°C)</label>
              <input
                type="number"
                value={editableParams.temperature || ''}
                onChange={(e) => updateParam('temperature', parseFloat(e.target.value))}
                className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">강수량 (mm)</label>
              <input
                type="number"
                value={editableParams.precipitation || ''}
                onChange={(e) => updateParam('precipitation', parseFloat(e.target.value))}
                className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">시간대</label>
              <div className="space-y-1">
                {['아침', '오전', '점심', '오후', '저녁', '야간'].map(slot => (
                  <label key={slot} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={editableParams.time_slots?.includes(slot) || false}
                      onChange={(e) => {
                        const currentSlots = editableParams.time_slots || [];
                        if (e.target.checked) {
                          updateParam('time_slots', [...currentSlots, slot]);
                        } else {
                          updateParam('time_slots', currentSlots.filter(s => s !== slot));
                        }
                      }}
                      className="mr-2"
                    />
                    {slot}
                  </label>
                ))}
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className="w-full py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:bg-green-300 font-medium"
            >
              {isAnalyzing ? '분석 중...' : '🔍 방송편성 추천'}
            </button>
          </div>
        ) : (
          <div className="text-gray-500 text-center py-8">
            채팅에 질문을 입력하면<br />파라미터가 표시됩니다
          </div>
        )}
      </div>

      {/* 결과 패널 */}
      <div className="w-1/4 border rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-bold mb-4">📋 추천 결과</h3>
        
        {recommendations.length > 0 ? (
          <div className="space-y-4">
            <table className="w-full border-collapse border border-gray-300 bg-white rounded-lg overflow-hidden">
              <thead className="bg-gray-100">
                <tr>
                  <th className="py-2 px-2 border-r border-gray-300 text-left font-semibold text-sm">시간대</th>
                  <th className="py-2 px-2 border-r border-gray-300 text-left font-semibold text-sm">상품명</th>
                  <th className="py-2 px-2 text-right font-semibold text-sm">예상 매출</th>
                </tr>
              </thead>
              <tbody>
                {recommendations.map((rec, index) => (
                  <tr key={index} className="border-b">
                    <td className="py-2 px-2 border-r text-sm">{rec.time_slot}</td>
                    <td className="py-2 px-2 border-r text-sm">{rec.features.product_name}</td>
                    <td className="py-2 px-2 text-right text-sm">{Math.round(rec.predicted_sales).toLocaleString()}원</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-gray-500 text-center py-8">
            분석 버튼을 클릭하면<br />추천 결과가 표시됩니다
          </div>
        )}
      </div>
    </div>
  );
}

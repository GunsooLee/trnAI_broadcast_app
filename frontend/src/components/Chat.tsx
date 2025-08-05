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
  gender?: string;
  age_group?: string;
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
  const [editableParams, setEditableParams] = useState<ExtractedParams | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoadingParams) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoadingParams(true);
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

  const updateParam = (key: keyof ExtractedParams, value: string | number | string[] | null) => {
    if (!editableParams) return;
    setEditableParams(prev => prev ? { ...prev, [key]: value } : null);
  };

  return (
    <div className="w-full h-screen p-4 bg-gray-100">
      <div className="grid grid-cols-5 grid-rows-2 gap-4 h-full">
        {/* 2사분면: 채팅창 (좌상) */}
        <div className="col-start-1 row-start-1 col-span-2 flex flex-col border rounded-lg shadow-lg bg-white">
          <div className="bg-blue-50 p-3 border-b rounded-t-lg">
            <h3 className="text-lg font-semibold text-blue-800">💬 채팅</h3>
          </div>
          <div className="flex-1 p-4 overflow-y-auto">
            {messages.map((msg, index) => (
              <div key={index} className={`my-2 p-3 rounded-lg max-w-[80%] ${msg.role === 'user' ? 'bg-blue-100 text-right ml-auto' : 'bg-gray-100 text-left mr-auto'}`}>
                <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
              </div>
            ))}
          </div>
          <div className="p-4 border-t flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-200"
              placeholder="편성 질문을 입력하세요…"
              disabled={isLoadingParams}
            />
            <button
              onClick={handleSendMessage}
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-blue-300 font-medium"
              disabled={isLoadingParams}
            >
              {isLoadingParams ? '분석 중...' : '전송'}
            </button>
          </div>
        </div>

        {/* 1사분면 + 4사분면: 추천 결과 (우측 전체) */}
        <div className="col-start-3 col-span-3 row-start-1 row-span-2 flex flex-col border rounded-lg shadow-lg bg-white">
          <div className="bg-green-50 p-3 border-b rounded-t-lg">
            <h3 className="text-lg font-semibold text-green-800">📋 추천 결과 및 분석</h3>
          </div>
          <div className="flex-1 p-4 overflow-y-auto">
            {isAnalyzing ? (
              <div className="text-gray-500 text-center py-20">
                <div className="text-6xl mb-6 animate-spin">⚙️</div>
                <div className="text-lg">분석 중입니다...</div>
              </div>
            ) : recommendations.length > 0 ? (
              <div className="space-y-6">
                {/* 추천 결과 테이블 */}
                <div>
                  <h4 className="text-lg font-semibold text-gray-800 mb-3">🎯 추천 목록</h4>
                  <table className="w-full border-collapse border border-gray-300 bg-white rounded-lg overflow-hidden">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="py-3 px-4 border-r border-gray-300 text-left font-semibold">시간대</th>
                        <th className="py-3 px-4 border-r border-gray-300 text-left font-semibold">상품명</th>
                        <th className="py-3 px-4 text-right font-semibold">예상 매출</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recommendations.map((rec, index) => (
                        <tr key={index} className="border-b hover:bg-gray-50">
                          <td className="py-3 px-4 border-r font-medium text-blue-600">{rec.time_slot}</td>
                          <td className="py-3 px-4 border-r">{rec.features.product_name}</td>
                          <td className="py-3 px-4 text-right font-semibold text-green-600">
                            {Math.round(rec.predicted_sales).toLocaleString()}원
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* 추천 통계 */}
                <div className="bg-blue-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-blue-800 mb-3">📊 추천 통계</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-white p-3 rounded border">
                      <span className="text-gray-600 block">총 추천 수</span>
                      <span className="font-bold text-lg text-blue-600">{recommendations.length}개</span>
                    </div>
                    <div className="bg-white p-3 rounded border">
                      <span className="text-gray-600 block">평균 매출</span>
                      <span className="font-bold text-lg text-blue-600">
                        {Math.round(recommendations.reduce((sum, rec) => sum + rec.predicted_sales, 0) / recommendations.length).toLocaleString()}원
                      </span>
                    </div>
                    <div className="bg-white p-3 rounded border">
                      <span className="text-gray-600 block">최고 매출</span>
                      <span className="font-bold text-lg text-green-600">
                        {Math.round(Math.max(...recommendations.map(rec => rec.predicted_sales))).toLocaleString()}원
                      </span>
                    </div>
                    <div className="bg-white p-3 rounded border">
                      <span className="text-gray-600 block">총 예상 매출</span>
                      <span className="font-bold text-lg text-purple-600">
                        {Math.round(recommendations.reduce((sum, rec) => sum + rec.predicted_sales, 0)).toLocaleString()}원
                      </span>
                    </div>
                  </div>
                </div>

                {/* 시간대별 분포 */}
                <div className="bg-green-50 p-4 rounded-lg">
                  <h4 className="font-semibold text-green-800 mb-3">🎯 시간대별 매출 분포</h4>
                  <div className="space-y-3">
                    {recommendations.map((rec, index) => (
                      <div key={index} className="bg-white p-3 rounded border">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-medium text-gray-800">{rec.time_slot}</span>
                          <span className="text-green-600 font-bold">
                            {Math.round(rec.predicted_sales).toLocaleString()}원
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div 
                            className="bg-gradient-to-r from-green-400 to-green-600 h-3 rounded-full transition-all duration-500" 
                            style={{
                              width: `${(rec.predicted_sales / Math.max(...recommendations.map(r => r.predicted_sales))) * 100}%`
                            }}
                          ></div>
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          상품: {rec.features.product_name}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-500 text-center py-20 flex flex-col items-center justify-center h-full">
                <div className="text-6xl mb-6">📊</div>
                <div className="text-lg">분석 버튼을 클릭하면<br />추천 결과가 표시됩니다</div>
              </div>
            )}
          </div>
        </div>

        {/* 3사분면: 분석 파라미터 (좌하) */}
        <div className="col-start-1 row-start-2 col-span-2 flex flex-col bg-white border rounded-lg shadow-lg">
          <div className="flex items-center justify-between bg-orange-50 p-3 border-b rounded-t-lg">
            <h3 className="text-lg font-semibold text-orange-800">📊 분석 파라미터</h3>
            <button
              onClick={handleAnalyze}
              disabled={!editableParams || isAnalyzing}
              className="px-4 py-1.5 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:bg-green-300 font-semibold text-sm transition-colors"
            >
              {isAnalyzing ? '분석중...' : '방송 추천'}
            </button>
          </div>
          
          <div className="flex-1 p-4 overflow-y-auto">
            {isLoadingParams ? (
              <div className="text-gray-500 text-center py-12">
                <div className="text-4xl mb-4 animate-spin">⚙️</div>
                <div>파라미터를 추출하고 있습니다...</div>
              </div>
            ) : editableParams ? (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">날짜</label>
                  <input
                    type="date"
                    value={editableParams.date || ''}
                    onChange={(e) => updateParam('date', e.target.value)}
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">날씨</label>
                  <select
                    value={editableParams.weather || ''}
                    onChange={(e) => updateParam('weather', e.target.value)}
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="">선택</option>
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
                    value={editableParams.temperature !== null && editableParams.temperature !== undefined ? editableParams.temperature.toString() : ''}
                    onChange={(e) => updateParam('temperature', e.target.value === '' ? null : parseFloat(e.target.value))}
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">강수량 (mm)</label>
                  <input
                    type="number"
                    value={editableParams.precipitation !== null && editableParams.precipitation !== undefined ? editableParams.precipitation.toString() : '0'}
                    onChange={(e) => updateParam('precipitation', e.target.value === '' ? 0 : parseFloat(e.target.value))}
                    min="0"
                    step="0.1"
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>

                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">시간대</label>
                  <div className="grid grid-cols-3 gap-2">
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
                        <span className="text-sm">{slot}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">카테고리</label>
                  <select
                    value={editableParams.categories?.[0] || ''}
                    onChange={(e) => updateParam('categories', e.target.value ? [e.target.value] : [])}
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="">전체</option>
                    <option value="건강식품">건강식품</option>
                    <option value="화장품">화장품</option>
                    <option value="생활용품">생활용품</option>
                    <option value="패션">패션</option>
                    <option value="전자제품">전자제품</option>
                    <option value="식품">식품</option>
                    <option value="운동용품">운동용품</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">성별 타겟</label>
                  <select
                    value={editableParams.gender || ''}
                    onChange={(e) => updateParam('gender', e.target.value)}
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="">전체</option>
                    <option value="남성">남성</option>
                    <option value="여성">여성</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">연령대</label>
                  <select
                    value={editableParams.age_group || ''}
                    onChange={(e) => updateParam('age_group', e.target.value)}
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="">전체</option>
                    <option value="20대">20대</option>
                    <option value="30대">30대</option>
                    <option value="40대">40대</option>
                    <option value="50대">50대</option>
                    <option value="60대 이상">60대 이상</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">요일 타입</label>
                  <select
                    value={editableParams.day_type || ''}
                    onChange={(e) => updateParam('day_type', e.target.value)}
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  >
                    <option value="">자동</option>
                    <option value="평일">평일</option>
                    <option value="주말">주말</option>
                  </select>
                </div>

                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">키워드</label>
                  <input
                    type="text"
                    value={editableParams.keywords?.join(', ') || ''}
                    onChange={(e) => updateParam('keywords', e.target.value.split(',').map(k => k.trim()).filter(k => k))}
                    placeholder="예: 다이어트, 건강, 미용"
                    className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
                  />
                </div>
                
              </div>
            ) : (
              <div className="text-gray-500 text-center py-12 flex flex-col items-center justify-center h-full">
                <div className="text-4xl mb-4">⚙️</div>
                <div>채팅에 질문을 입력하면<br />파라미터가 표시됩니다</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

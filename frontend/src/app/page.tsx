import Chat from '@/components/Chat';

export default function Home() {
  return (
    <main className="min-h-screen p-4">
        <h1 className="text-3xl font-bold text-center mb-8">홈쇼핑 방송 추천 챗봇</h1>
        <Chat />
    </main>
  );
}

import Image from "next/image";
import Chat from '@/components/Chat';

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center p-4 sm:p-12 md:p-24">
      <div className="w-full max-w-3xl">
        <h1 className="text-3xl font-bold text-center mb-8">홈쇼핑 방송 추천 챗봇</h1>
        <Chat />
      </div>
    </main>
  );
}

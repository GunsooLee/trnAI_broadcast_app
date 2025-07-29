/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://backend:8501/api/:path*',
      },
    ];
  },
};

export default nextConfig;

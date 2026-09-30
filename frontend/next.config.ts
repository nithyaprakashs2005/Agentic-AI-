import type { NextConfig } from "next";

const API_BASE = process.env.API_BASE_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  agentRules: false,
  experimental: {
    // The rewrite proxy defaults to 30s and then drops the socket, which the
    // browser sees as a 500 while the backend goes on to answer normally. An
    // agent turn is several sequential model calls -- measured at 8s on a good
    // run and 67s on a bad one -- so 30s cut off real, successful work.
    //
    // This is a ceiling for pathological cases, not a target: the per-request
    // deadline in lib/api.ts is what actually governs how long a user waits.
    proxyTimeout: 180_000,
  },
  // Proxy the API through Next so the browser stays same-origin: no CORS
  // preflight on every turn, and SSE is not subject to cross-origin buffering.
  async rewrites() {
    return [
      { source: "/api/backend/:path*", destination: `${API_BASE}/api/:path*` },
      { source: "/api/health/:path*", destination: `${API_BASE}/health/:path*` },
      { source: "/api/health", destination: `${API_BASE}/health` },
    ];
  },
};

export default nextConfig;

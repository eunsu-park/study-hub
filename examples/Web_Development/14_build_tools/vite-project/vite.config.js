/**
 * Vite Configuration File
 * https://vitejs.dev/config/
 */

import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
    // Dev server configuration
    server: {
        port: 3000,           // Port number
        open: true,           // Auto-open browser
        cors: true,           // Allow CORS
        host: true,           // Allow network access
        // Proxy configuration (for API server integration)
        // proxy: {
        //     '/api': {
        //         target: 'http://localhost:8080',
        //         changeOrigin: true,
        //         rewrite: (path) => path.replace(/^\/api/, '')
        //     }
        // }
    },

    // Why: Separating build options from dev server config keeps concerns isolated and
    // lets CI pipelines override build settings without touching dev defaults
    // Build configuration
    build: {
        outDir: 'dist',       // Output directory
        sourcemap: true,      // Generate source maps
        minify: 'terser',     // Minification method (terser or esbuild)
        target: 'es2020',     // Target browser

        // Bundle splitting configuration
        rollupOptions: {
            input: {
                main: resolve(__dirname, 'index.html'),
                // For multi-page apps
                // about: resolve(__dirname, 'about.html'),
            },
            output: {
                // Chunk filename pattern
                chunkFileNames: 'assets/js/[name]-[hash].js',
                entryFileNames: 'assets/js/[name]-[hash].js',
                assetFileNames: 'assets/[ext]/[name]-[hash].[ext]',

                // Vendor bundle separation
                manualChunks: {
                    // vendor: ['lodash', 'axios'],
                }
            }
        },

        // Chunk size warning threshold (KB)
        chunkSizeWarningLimit: 500,
    },

    // Why: Path aliases eliminate fragile relative imports (../../components/foo) and let you
    // reorganize directory structure without updating import paths across the entire codebase
    // Path alias configuration
    resolve: {
        alias: {
            '@': resolve(__dirname, 'src'),
            '@components': resolve(__dirname, 'src/components'),
            '@utils': resolve(__dirname, 'src/utils'),
            '@styles': resolve(__dirname, 'src/styles'),
        }
    },

    // CSS configuration
    css: {
        // CSS modules configuration
        modules: {
            localsConvention: 'camelCase',
        },
        // PostCSS configuration
        postcss: {
            plugins: [
                // Add plugins like autoprefixer
            ]
        },
        // Preprocessor options
        preprocessorOptions: {
            scss: {
                // Auto-import global variables file
                // additionalData: `@import "@/styles/variables.scss";`
            }
        }
    },

    // Why: The VITE_ prefix acts as a safeguard - only prefixed env vars are exposed to client
    // code, preventing accidental leakage of server-side secrets (DB passwords, API keys) into bundles
    // Environment variable prefix
    envPrefix: 'VITE_',

    // Plugins
    plugins: [
        // @vitejs/plugin-react
        // @vitejs/plugin-vue
        // @vitejs/plugin-legacy (legacy browser support)
    ],

    // Optimization configuration
    optimizeDeps: {
        // Dependencies to pre-bundle
        include: [],
        // Dependencies to exclude from pre-bundling
        exclude: []
    },

    // Preview server configuration
    preview: {
        port: 4173,
        open: true,
    }
});

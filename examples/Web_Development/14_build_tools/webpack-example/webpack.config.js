/**
 * Webpack Configuration File
 * https://webpack.js.org/configuration/
 */

const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
    const isProduction = argv.mode === 'production';

    return {
        // Entry point
        entry: {
            main: './src/index.js',
            // Multiple entry point example
            // vendor: './src/vendor.js',
        },

        // Output configuration
        // Why: contenthash in production filenames enables aggressive browser caching - when code changes,
        // the hash changes, busting the cache; in dev, plain names enable easier debugging
        output: {
            path: path.resolve(__dirname, 'dist'),
            filename: isProduction ? '[name].[contenthash].js' : '[name].js',
            chunkFilename: isProduction ? '[name].[contenthash].chunk.js' : '[name].chunk.js',
            clean: true,  // Clean dist folder before build
            publicPath: '/',
        },

        // Dev server configuration
        devServer: {
            static: {
                directory: path.join(__dirname, 'public'),
            },
            port: 3000,
            open: true,
            hot: true,  // Enable HMR
            // Why: historyApiFallback redirects all 404s to index.html, enabling client-side
            // routing in SPAs where the server doesn't know about frontend routes
            historyApiFallback: true,  // SPA routing support
            compress: true,
            // Proxy configuration
            // proxy: {
            //     '/api': {
            //         target: 'http://localhost:8080',
            //         changeOrigin: true,
            //     }
            // }
        },

        // Module loader configuration
        module: {
            rules: [
                // JavaScript/JSX processing
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-env'],
                            cacheDirectory: true,
                        }
                    }
                },

                // CSS processing
                {
                    test: /\.css$/,
                    use: [
                        isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
                        {
                            loader: 'css-loader',
                            options: {
                                sourceMap: !isProduction,
                            }
                        }
                    ]
                },

                // Why: 'asset' type auto-decides between inline (data URL) and file based on size -
                // small images inline to save HTTP requests, large ones stay as files to avoid bloating JS bundles
                // Image processing (asset modules)
                {
                    test: /\.(png|jpe?g|gif|svg|webp)$/i,
                    type: 'asset',
                    parser: {
                        dataUrlCondition: {
                            maxSize: 8 * 1024, // Inline if 8KB or less
                        }
                    },
                    generator: {
                        filename: 'images/[name].[hash:8][ext]'
                    }
                },

                // Font processing
                {
                    test: /\.(woff|woff2|eot|ttf|otf)$/i,
                    type: 'asset/resource',
                    generator: {
                        filename: 'fonts/[name].[hash:8][ext]'
                    }
                }
            ]
        },

        // Plugins
        plugins: [
            // HTML template processing
            new HtmlWebpackPlugin({
                template: './src/index.html',
                filename: 'index.html',
                inject: 'body',
                minify: isProduction ? {
                    removeComments: true,
                    collapseWhitespace: true,
                    removeAttributeQuotes: true,
                } : false,
            }),

            // CSS extraction (production)
            ...(isProduction ? [
                new MiniCssExtractPlugin({
                    filename: 'css/[name].[contenthash].css',
                    chunkFilename: 'css/[name].[contenthash].chunk.css',
                })
            ] : []),
        ],

        // Path aliases
        resolve: {
            extensions: ['.js', '.json'],
            alias: {
                '@': path.resolve(__dirname, 'src'),
                '@components': path.resolve(__dirname, 'src/components'),
                '@utils': path.resolve(__dirname, 'src/utils'),
                '@styles': path.resolve(__dirname, 'src/styles'),
            }
        },

        // Optimization
        optimization: {
            // Code splitting
            splitChunks: {
                chunks: 'all',
                cacheGroups: {
                    // Vendor bundle separation
                    vendors: {
                        test: /[\\/]node_modules[\\/]/,
                        name: 'vendors',
                        priority: -10,
                    },
                    // Common module separation
                    common: {
                        minChunks: 2,
                        priority: -20,
                        reuseExistingChunk: true,
                    }
                }
            },
            // Why: Extracting the runtime into a separate chunk prevents its hash from changing
            // in all entry bundles when only one module changes, preserving cache validity
            // Runtime chunk separation
            runtimeChunk: 'single',
        },

        // Source maps
        devtool: isProduction ? 'source-map' : 'eval-source-map',

        // Performance hints
        performance: {
            hints: isProduction ? 'warning' : false,
            maxEntrypointSize: 250000,  // 250KB
            maxAssetSize: 250000,
        },

        // Why: Filesystem caching persists build results to disk between runs, cutting
        // rebuild times dramatically for large projects (often from minutes to seconds)
        // Cache configuration
        cache: {
            type: 'filesystem',
            buildDependencies: {
                config: [__filename],
            }
        },

        // Stats output configuration
        stats: {
            colors: true,
            modules: false,
            children: false,
        }
    };
};

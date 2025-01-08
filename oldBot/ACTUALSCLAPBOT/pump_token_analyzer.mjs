import WebSocket from 'ws';
import fs from 'fs';
import path from 'path';
import axios from 'axios';




class Logger {
    constructor(logDirectory) {
        this.logDirectory = logDirectory;
        // Ensure log directory exists
        if (!fs.existsSync(logDirectory)) {
            fs.mkdirSync(logDirectory, { recursive: true });
        }
    }

    log(level, message) {
        const timestamp = new Date().toISOString();
        const logMessage = `${timestamp} [${level.toUpperCase()}]: ${message}\n`;
        
        // File logging
        const logFileName = `${level}-${new Date().toISOString().split('T')[0]}.log`;
        const logFilePath = path.join(this.logDirectory, logFileName);
        
        try {
            fs.appendFileSync(logFilePath, logMessage);
        } catch (error) {
            console.error(`Failed to write to log file: ${error}`);
        }
    }
}

class PumpTokenAnalyzer {
    static GOOD_BUY_THRESHOLD = 0.40;   // 60% increase
    static BAD_BUY_THRESHOLD = -0.18;   // 15% decrease
    static RUG_THRESHOLD = -0.5;       // 50% decrease
    static EXCEPTIONAL_GAIN = 1.2;     // 120% total gain

    constructor(config) {
        this.ws = null;
        this.tokenData = new Map();
        this.config = config;
        this.logger = new Logger(config.logDirectory);
        this.transactionHistory = [];
    }

    // Replace the fetchCoinDetails method
async fetchCoinDetails(mintStr) {
    try {
        const url = `https://frontend-api.pump.fun/coins/${mintStr}`;
        const response = await axios.get(url, {
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.pump.fun/",
                "Origin": "https://www.pump.fun",
                "Connection": "keep-alive",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "cross-site"
            },
            timeout: 5000 // Add timeout to prevent hanging
        });
        
        if (response.data) {
            return {
                name: response.data.name || '',
                description: response.data.description || '',
                twitter: response.data.twitter || false,
                telegram: response.data.telegram || false,
                website: response.data.website || false,
                reply_count: response.data.reply_count || 0,
                king_of_the_hill_timestamp: response.data.king_of_the_hill_timestamp
            };
        }
        return null;
    } catch (error) {
        this.logger.log('error', `Error fetching coin details for ${mintStr}: ${error.message}`);
        return null;
    }
}

// Replace the analyzeSocialSignals function
analyzeSocialSignals(coinDetails) {
    if (!coinDetails) {
        return {
            has_twitter: false,
            has_telegram: false,
            has_website: false,
            description_length: 0,
            reply_count: 0
        };
    }

    const result = {
        has_twitter: Boolean(coinDetails.twitter),
        has_telegram: Boolean(coinDetails.telegram),
        has_website: Boolean(coinDetails.website),
        description_length: (coinDetails.description || '').length,
        reply_count: Number(coinDetails.reply_count) || 0
    };

    return result;
}

    connect() {
        this.logger.log('info', `Attempting to connect to WebSocket: ${this.config.websocketUrl}`);
        
        try {
            this.ws = new WebSocket(this.config.websocketUrl);

            this.ws.on('open', () => {
                this.logger.log('info', 'Connected to WebSocket');
                
                // Subscribe to all events
                const subscriptions = [
                    { method: "subscribeNewToken" }
                ];

                subscriptions.forEach(sub => {
                    this.ws?.send(JSON.stringify(sub));
                    this.logger.log('info', `Subscribed to ${sub.method}`);
                });
            });

            this.ws.on('message', this.processMessage.bind(this));

            this.ws.on('error', (error) => {
                this.logger.log('error', `WebSocket error: ${error.message}`);
            });

            this.ws.on('close', () => {
                this.logger.log('warn', 'Connection closed, attempting to reconnect in 5 seconds...');
                setTimeout(() => this.connect(), 5000);
            });

            // Set up periodic token analysis
            setInterval(async () => await this.analyzeTokens(), this.config.analyzeInterval);

        } catch (error) {
            this.logger.log('error', `Connection setup error: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
    }

    processMessage(data) {
        try {
            const message = JSON.parse(data.toString());
            
            // Log the full message for debugging
            this.logger.log('debug', `Full message: ${JSON.stringify(message)}`);

            // Determine message type and handle accordingly
            switch(message.txType) {
                case 'create':
                    this.handleTokenCreation(message);
                    break;
                case 'buy':
                    this.handleBuyTransaction(message);
                    break;
                case 'sell':
                    this.handleSellTransaction(message);
                    break;
                default:
                    this.logger.log('warn', `Unhandled message type: ${JSON.stringify(message)}`);
            }

            // Store transaction in history
            this.transactionHistory.push(message);
            // Optionally, limit history size
            if (this.transactionHistory.length > 1000) {
                this.transactionHistory.shift();
            }
        } catch (error) {
            this.logger.log('error', `Message processing error: ${error instanceof Error ? error.message : 'Unknown error'}`);
            this.logger.log('debug', `Raw message: ${data.toString()}`);
        }
    }

    async handleTokenCreation(creationData) {
        // Create token data entry with minimal initial metrics
        const tokenEntry = {
            createdAt: new Date(),
            firstTradeTime: null,  // Will be set on first trade
            details: creationData,
            transactions: [],
            metrics: {
                initialMarketCap: null,    // Will be set on first trade
                currentMarketCap: null,    // Will be set on first trade
                highestMarketCap: null,    // Will be set on first trade
                lowestMarketCap: null,     // Will be set on first trade
                totalVolume: 0,
                hasHitGoodBuy: false,
                hasHitBadBuy: false,
                timeToRug: null,
                recordWritten: false,
                goodTradeConfirmationTime: null  // Time when we should confirm a good trade
            }
        };

        this.tokenData.set(creationData.mint, tokenEntry);

        this.logger.log('info', `New Token Created: 
            Mint: ${creationData.mint}`);

        if (this.ws) {
            const tradeSub = {
                method: "subscribeTokenTrade",
                keys: [creationData.mint]
            };
            this.ws.send(JSON.stringify(tradeSub));
        }
    }

    initializeMetricsOnFirstTrade(token, marketCap) {
        if (!token.firstTradeTime) {
            token.firstTradeTime = new Date();
            token.metrics.initialMarketCap = marketCap;
            token.metrics.currentMarketCap = marketCap;
            token.metrics.highestMarketCap = marketCap;
            token.metrics.lowestMarketCap = marketCap;
            
            this.logger.log('debug', `Initialized metrics for first trade:
                Initial Market Cap: ${marketCap}
                First Trade Time: ${token.firstTradeTime.toISOString()}`);
        }
    }

    updateMarketCapMetrics(token, newMarketCap) {
        // Initialize metrics on first trade
        this.initializeMetricsOnFirstTrade(token, newMarketCap);
    
        // Update market cap tracking
        token.metrics.currentMarketCap = newMarketCap;
        token.metrics.highestMarketCap = Math.max(token.metrics.highestMarketCap, newMarketCap);
        token.metrics.lowestMarketCap = Math.min(token.metrics.lowestMarketCap, newMarketCap);
    
        const returnFromInitial = ((newMarketCap - token.metrics.initialMarketCap) / token.metrics.initialMarketCap) * 100;
        const drawdownFromPeak = ((token.metrics.highestMarketCap - newMarketCap) / token.metrics.highestMarketCap) * 100;
    
        // Check for good trade condition (50% increase at any point)
        if (returnFromInitial >= PumpTokenAnalyzer.GOOD_BUY_THRESHOLD * 100 && !token.metrics.goodTradeConfirmationTime) {
            token.metrics.hasHitGoodBuy = true;
            token.metrics.goodTradeConfirmationTime = Date.now() + 300000; // Set confirmation time 1 minute in future
        }
    
        // Check for rug condition: 
        // 1. Must have first hit a good trade (50% gain)
        // 2. Then drop below initial market cap
        if (token.metrics.hasHitGoodBuy && 
            drawdownFromPeak >= Math.abs(PumpTokenAnalyzer.RUG_THRESHOLD * 100) && 
            token.metrics.timeToRug === null) {
            // Calculate time to rug from first trade
            token.metrics.timeToRug = (new Date() - token.firstTradeTime) / 1000;
            token.metrics.hasRugged = true;
        }

        // Check for bad trade condition (25% drop before any 50% gain)
        if (!token.metrics.hasHitGoodBuy && drawdownFromPeak >= Math.abs(PumpTokenAnalyzer.BAD_BUY_THRESHOLD * 100)) {
            token.metrics.hasHitBadBuy = true;
        }
    }

    async handleBuyTransaction(buyData) {
        const token = this.tokenData.get(buyData.mint);

        if (token) {
            token.transactions.push({
                type: 'buy',
                timestamp: new Date().toISOString(),
                ...buyData
            });

            // Update token metrics
            token.metrics.totalVolume += buyData.solAmount;
            this.updateMarketCapMetrics(token, buyData.marketCapSol);

            this.logger.log('info', `Buy Transaction: 
                Token: ${buyData.mint}
                SOL Amount: ${buyData.solAmount}
                Market Cap: ${buyData.marketCapSol}`);
        } else {
            this.logger.log('warn', `Buy transaction for unknown token: ${buyData.mint}`);
        }
    }

    async handleSellTransaction(sellData) {
        const token = this.tokenData.get(sellData.mint);

        if (token) {
            token.transactions.push({
                type: 'sell',
                timestamp: new Date().toISOString(),
                ...sellData
            });

            // Update token metrics
            token.metrics.totalVolume += sellData.solAmount;
            this.updateMarketCapMetrics(token, sellData.marketCapSol);

            this.logger.log('info', `Sell Transaction: 
                Token: ${sellData.mint}
                SOL Amount: ${sellData.solAmount}
                Market Cap: ${sellData.marketCapSol}`);
        } else {
            this.logger.log('warn', `Sell transaction for unknown token: ${sellData.mint}`);
        }
    }

    shouldWriteAnalysis(tokenData, currentTime) {
        // Bad trades can be written immediately
        if (tokenData.metrics.hasHitBadBuy && !tokenData.metrics.hasHitGoodBuy && !tokenData.metrics.recordWritten) {
            return true;
        }

        // Good trades need to wait for confirmation period
        if (tokenData.metrics.hasHitGoodBuy &&
            tokenData.metrics.goodTradeConfirmationTime &&
            currentTime >= tokenData.metrics.goodTradeConfirmationTime &&
            !tokenData.metrics.recordWritten) {
            return true;
        }

        // Exceptional gains also need confirmation period
        const currentReturn = ((tokenData.metrics.currentMarketCap - tokenData.metrics.initialMarketCap) / tokenData.metrics.initialMarketCap) * 100;
        if (currentReturn >= 80 &&
            tokenData.metrics.goodTradeConfirmationTime &&
            currentTime >= tokenData.metrics.goodTradeConfirmationTime &&
            !tokenData.metrics.recordWritten) {
            return true;
        }

        return false;
    }

    async analyzeTokens() {
        for (const [tokenAddress, tokenData] of this.tokenData.entries()) {
            // Skip tokens with no trades or no initialized metrics
            if (!tokenData.firstTradeTime || tokenData.transactions.length === 0) continue;

            const currentTime = Date.now();
            const timeSinceFirstTrade = (currentTime - tokenData.firstTradeTime) / 1000;

            const buys = tokenData.transactions.filter(t => t.type === 'buy');
            const sells = tokenData.transactions.filter(t => t.type === 'sell');

            // Calculate current state
            const currentMarketCap = tokenData.metrics.currentMarketCap;
            const initialMarketCap = tokenData.metrics.initialMarketCap;
            const peakMarketCap = tokenData.metrics.highestMarketCap;

            // Calculate returns
            const currentReturn = ((currentMarketCap - initialMarketCap) / initialMarketCap) * 100;
            const peakReturn = ((peakMarketCap - initialMarketCap) / initialMarketCap) * 100;
            const drawdownFromPeak = ((peakMarketCap - currentMarketCap) / peakMarketCap) * 100;

            // Enhanced trade window diagnostic logging
            
            // Modify getTradesInWindow to be more robust
            const getTradesInWindow = (trades, firstTradeTime, startSeconds, endSeconds) => {
                if (!firstTradeTime || trades.length === 0) return [];

                // Ensure firstTradeTime is a Date object
                const tradeFirstTime = firstTradeTime instanceof Date
                    ? firstTradeTime
                    : new Date(firstTradeTime);

                return trades.filter(t => {
                    // Ensure timestamp is a valid date
                    const tradeTimestamp = new Date(t.timestamp);

                    // Calculate trade time in seconds since first trade
                    const tradeTime = (tradeTimestamp - tradeFirstTime) / 1000;

                    // Return trades within the specified time window
                    return tradeTime >= startSeconds && tradeTime < endSeconds;
                });
            };


     
            // Technical Indicators Calculation
            const calculateTechnicalIndicators = (trades) => {
                if (trades.length === 0) return {};
     
                const prices = trades.map(t => t.marketCapSol);
                const volumes = trades.map(t => t.solAmount);
     
                // Simple Relative Strength Index (RSI) approximation
                const priceChanges = prices.slice(1).map((price, i) => price - prices[i]);
                const gains = priceChanges.filter(change => change > 0);
                const losses = priceChanges.filter(change => change < 0).map(Math.abs);
                
                const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / gains.length : 0;
                const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / losses.length : 0;
                const rs = avgLoss > 0 ? avgGain / avgLoss : 0;
                const rsi = 100 - (100 / (1 + rs));
     
                return {
                    rsi: rsi,
                    price_volatility: calculateVolatility(prices),
                    volume_volatility: calculateVolatility(volumes),
                    momentum: calculateMomentum(prices)
                };
            };
     
            // Utility Functions for Technical Indicators
            const calculateVolatility = (values) => {
                if (values.length < 2) return 0;
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
                return Math.sqrt(variance);
            };
     
            const calculateMomentum = (prices) => {
                if (prices.length < 2) return 0;
                return (prices[prices.length - 1] - prices[0]) / prices[0] * 100;
            };
     
            // Social Signal Analysis
            const analyzeSocialSignals = (coinDetails) => {
                if (!coinDetails) return {
                    has_twitter: false,
                    has_telegram: false,
                    has_website: false,
                    description_length: 0,
                    reply_count: 0
                };
     
                return {
                    has_twitter: !!coinDetails.twitter,
                    has_telegram: !!coinDetails.telegram,
                    has_website: !!coinDetails.website,
                    description_length: (coinDetails.description || '').length,
                    reply_count: coinDetails.reply_count || 0
                };
            };
     
            // Trading Dynamics Analysis
            const analyzeTradePatterns = (trades) => {
                if (trades.length < 2) return {
                    trade_amount_variance: 0,
                    avg_time_between_trades: null,
                    unique_wallets: 0,
                    trade_concentration: 0
                };
            
                const tradeAmounts = trades.map(t => t.solAmount);
                const timeBetweenTrades = trades.slice(1).map((trade, i) => 
                    (new Date(trade.timestamp) - new Date(trades[i].timestamp)) / 1000
                );
            
                // Use traderPublicKey instead of other wallet fields
                const uniqueWallets = new Set();
                trades.forEach(trade => {
                    if (trade.traderPublicKey) {
                        uniqueWallets.add(trade.traderPublicKey);
                    }
                });
            
                // Track volume by wallet using traderPublicKey
                const tradesByWallet = {};
                trades.forEach(trade => {
                    if (trade.traderPublicKey) {
                        tradesByWallet[trade.traderPublicKey] = (tradesByWallet[trade.traderPublicKey] || 0) + trade.solAmount;
                    }
                });
                
                const sortedTrades = Object.values(tradesByWallet).sort((a, b) => b - a);
                const totalVolume = sortedTrades.reduce((a, b) => a + b, 0);
                
                // Gini coefficient calculation
                const n = sortedTrades.length;
                if (n === 0) return {
                    trade_amount_variance: 0,
                    avg_time_between_trades: null,
                    unique_wallets: 0,
                    trade_concentration: 0
                };
            
                const cumulative = sortedTrades.reduce((acc, val, idx) => {
                    const prev = acc[idx] || 0;
                    const cumVal = prev + (val / totalVolume);
                    acc.push(cumVal);
                    return acc;
                }, [0]);
            
                const sumOfCumulative = cumulative.reduce((a, b) => a + b, 0);
                const giniCoefficient = (n + 1 - 2 * sumOfCumulative / n) / n;
            
                return {
                    trade_amount_variance: calculateVolatility(tradeAmounts),
                    avg_time_between_trades: timeBetweenTrades.length > 0 ?
                        timeBetweenTrades.reduce((a, b) => a + b, 0) / timeBetweenTrades.length : null,
                    unique_wallets: uniqueWallets.size,
                    trade_concentration: giniCoefficient
                };
            };
     
            // Trade Concentration Analysis
            const calculateTradeConcentration = (trades) => {
                const tradesByWallet = {};
                trades.forEach(trade => {
                    tradesByWallet[trade.wallet] = (tradesByWallet[trade.wallet] || 0) + trade.solAmount;
                });
                
                const sortedTrades = Object.values(tradesByWallet).sort((a, b) => b - a);
                const totalVolume = sortedTrades.reduce((a, b) => a + b, 0);
                
                const n = sortedTrades.length;
                const cumulative = sortedTrades.reduce((acc, val, idx) => {
                    const prev = acc[idx];
                    const cumVal = prev + (val / totalVolume);
                    acc.push(cumVal);
                    return acc;
                }, [0]);
     
                const sumOfCumulative = cumulative.reduce((a, b) => a + b, 0);
                return (n + 1 - 2 * sumOfCumulative / n) / n;
            };
     
           // Detailed trade window analysis
// Detailed trade window analysis
const tradeWindows = [
    { start: 0, end: 60, label: 'first_minute' },
    { start: 0, end: 10, label: 'first_10_seconds' },
    { start: 0, end: 20, label: 'first_20_seconds' },
    { start: 0, end: 30, label: 'first_30_seconds' },
    { start: 30, end: 40, label: '30_to_40_seconds' },
    { start: 40, end: 50, label: '40_to_50_seconds' },
    { start: 50, end: 60, label: '50_to_60_seconds' },
    { start: 60, end: 120, label: '1_to_2_minutes' },
    { start: 120, end: 180, label: '2_to_3_minutes' }
];



const tradeWindowDetails = tradeWindows.reduce((acc, window) => {
    const windowTrades = getTradesInWindow(
        tokenData.transactions, 
        tokenData.firstTradeTime,
        window.start, 
        window.end
    );
    
    const windowBuys = windowTrades.filter(t => t.type === 'buy');
    const windowSells = windowTrades.filter(t => t.type === 'sell');

    const windowDuration = window.end - window.start;

    acc[`${window.label}_total_trades`] = windowTrades.length;
    acc[`${window.label}_buy_count`] = windowBuys.length;
    acc[`${window.label}_sell_count`] = windowSells.length;
    acc[`${window.label}_total_volume`] = windowTrades.reduce((sum, t) => sum + (t.solAmount || 0), 0);
    acc[`${window.label}_buy_volume`] = windowBuys.reduce((sum, t) => sum + (t.solAmount || 0), 0);
    acc[`${window.label}_sell_volume`] = windowSells.reduce((sum, t) => sum + (t.solAmount || 0), 0);
    acc[`${window.label}_transaction_rate`] = windowTrades.length / windowDuration;

    return acc;
}, {});
     
            // Determine if we should write the record
            let shouldWrite = false;
            let finalClassification = 'neutral';
     
            // Case 1: Rugged trade - always write
            if (tokenData.metrics.hasRugged && !tokenData.metrics.recordWritten) {
                shouldWrite = true;
                finalClassification = 'good';
            }
            // Case 2: Bad trade (25% drop before 50% gain)
            else if (tokenData.metrics.hasHitBadBuy && !tokenData.metrics.hasHitGoodBuy && !tokenData.metrics.recordWritten) {
                shouldWrite = true;
                finalClassification = 'bad';
            }
            // Case 3: Good token that has not rugged - only write after 3 minutes
            else if (tokenData.metrics.hasHitGoodBuy && 
                     timeSinceFirstTrade >= 180 && 
                     !tokenData.metrics.hasRugged && 
                     !tokenData.metrics.recordWritten) {
                shouldWrite = true;
                finalClassification = 'good';
            }
            // Case 4: Neutral trade (stayed within bounds for 3 minutes)
            else if (timeSinceFirstTrade >= 180 && 
                     !tokenData.metrics.hasHitGoodBuy && 
                     !tokenData.metrics.hasHitBadBuy && 
                     !tokenData.metrics.recordWritten) {
                shouldWrite = true;
                finalClassification = 'neutral';
            }
     
            if (shouldWrite) {
                // Fetch coin details only when we're about to write the record
                let coinDetails = null;
                try {
                    coinDetails = await this.fetchCoinDetails(tokenAddress);
                } catch (error) {
                    this.logger.log('error', `Failed to fetch final coin details for ${tokenAddress}: ${error}`);
                }
     
                // Calculate final metrics
                const last5Transactions = tokenData.transactions.slice(-5);
                const volumeMA5 = last5Transactions.reduce((sum, t) => sum + (t.solAmount || 0), 0) / 5;
     
                const tokenAnalysis = {
                    ...tradeWindowDetails,
                    
                    // Technical Indicators
                    technical_indicators: calculateTechnicalIndicators(tokenData.transactions),
                    
                    // Social Signals
                    social_signals: this.analyzeSocialSignals(coinDetails),
                    
                    // Trading Dynamics
                    trade_patterns: analyzeTradePatterns(tokenData.transactions),
     
                    // Token Metadata
                    token_metadata: {
                        name_length: (coinDetails?.name || '').length,
                        description_length: (coinDetails?.description || '').length
                    },
     
                    // Market Context
                    market_context: {
                        creation_time: tokenData.firstTradeTime.toISOString(),
                    },
     
                    timestamp: new Date(currentTime).toISOString(),
                    tokenAddress: tokenAddress,
                    market_cap: currentMarketCap,
                    initial_market_cap: initialMarketCap,
                    peak_market_cap: peakMarketCap,
                    initial_investment_ratio: currentMarketCap / initialMarketCap,
                    volume: tokenData.metrics.totalVolume,
                    volume_ma5: volumeMA5,
                    volume_change: tokenData.transactions.length > 1 ? 
                        tokenData.transactions[tokenData.transactions.length - 1].solAmount / 
                        tokenData.transactions[tokenData.transactions.length - 2].solAmount - 1 : 0,
                    transaction_count: tokenData.transactions.length,
                    buy_count: buys.length,
                    sell_count: sells.length,
                    buy_sell_ratio: sells.length > 0 ? buys.length / sells.length : buys.length,
                    market_cap_change: currentReturn / 100,
                    peak_return: peakReturn / 100,
                    seconds_since_creation: timeSinceFirstTrade,
                    transaction_rate: tokenData.transactions.length / (timeSinceFirstTrade / 5),
                    buy_pressure: buys.slice(-10).length,
                    sell_pressure: sells.slice(-10).length,
     
                    // Classification metrics
                    buy_classification: finalClassification,
                    time_to_rug: tokenData.metrics.timeToRug,
                    has_rugged: tokenData.metrics.hasRugged,
                    highest_market_cap: tokenData.metrics.highestMarketCap,
                    lowest_market_cap: tokenData.metrics.lowestMarketCap,
                    max_drawdown: drawdownFromPeak / 100,
     
                    // Coin details
                    coin_name: coinDetails?.name || "",
                    description: coinDetails?.description || "",
                    twitter: coinDetails?.twitter || false,
                    telegram: coinDetails?.telegram || false,
                    website: coinDetails?.website || false,
                    reply_count: coinDetails?.reply_count || 0,
                    king_of_the_hill_status: coinDetails?.king_of_the_hill_timestamp !== null
                };
        
                const analysisFilePath = path.join(
                    this.config.logDirectory, 
                    `new-data.log`
                );
        
                try {
                    fs.mkdirSync(this.config.logDirectory, { recursive: true });
                    const logMessage = `${new Date(currentTime).toISOString()} [FINAL-ANALYSIS]: ${JSON.stringify(tokenAnalysis)}\n`;
                    fs.appendFileSync(analysisFilePath, logMessage, 'utf8');
                    console.log(`Appended final analysis for token ${tokenAddress} (${finalClassification})`);
                    
                    // Mark that we've written a record for this token
                    tokenData.metrics.recordWritten = true;
                } catch (error) {
                    console.error(`Error writing analysis for token ${tokenAddress}:`, error);
                }
            }
        }
     }

    getTransactionHistory() {
        return this.transactionHistory;
    }

    getTokenData(mint) {
        return this.tokenData.get(mint);
    }
}

// Configuration
const config = {
    websocketUrl: 'wss://pumpportal.fun/api/data',
    logDirectory: './logs',
    analyzeInterval: 3000
};

// Run the analyzer
const analyzer = new PumpTokenAnalyzer(config);
analyzer.connect();

export default PumpTokenAnalyzer;

// Additional utility function for trade execution (placeholder)
export async function executeTokenTrade(
    connection, 
    wallet, 
    mint, 
    action, 
    amount, 
    denominatedInSol = true
) {
    return null;
}
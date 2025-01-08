import express from 'express';
import { VersionedTransaction, Connection, Keypair, PublicKey } from '@solana/web3.js';
import bs58 from 'bs58';
import { TOKEN_PROGRAM_ID } from '@solana/spl-token';

const app = express();
app.use(express.json());

const RPC_ENDPOINT = "https://api.mainnet-beta.solana.com";
const web3Connection = new Connection(RPC_ENDPOINT, 'confirmed');
const PRIVATE_KEY = ""
const PUBLIC_KEY = "6qdXtZFwJWXW7kWSbU6Zphw2DW9xbiCZJY4XqqpKSURM"

// Enhanced logging function
function enhancedLog(message, data) {
    console.log(`[${new Date().toISOString()}] ${message}`);
    if (data) {
        try {
            console.log(JSON.stringify(data, null, 2));
        } catch (error) {
            console.log(data);
        }
    }
}

// List of pools to try
const POOLS = ['pump', 'raydium'];

app.post('/trade', async (req, res) => {
    try {
        const { mint, action, amount, pool: requestedPool } = req.body;
        
        enhancedLog('Received trade request', { mint, action, amount, requestedPool });
        
        let tradeAmount = amount;
        let denominatedInSol = "true";
        let slippage = 50;

        // If selling, get current token balance
        if (action === 'sell') {
            denominatedInSol = "false";
            slippage = 99
            tradeAmount = "100%"
        }

        // Determine pools to try
        const poolsToTry = requestedPool 
            ? [requestedPool, ...POOLS.filter(p => p !== requestedPool)] 
            : POOLS;

        let lastError = null;

        // Try trading through multiple pools
        for (const pool of poolsToTry) {
            try {
                const requestBody = {
                    "publicKey": PUBLIC_KEY,
                    "action": action,
                    "mint": mint,
                    "denominatedInSol": denominatedInSol,
                    "amount": tradeAmount,
                    "slippage": slippage,
                    "priorityFee": 0.003,
                    "pool": pool
                };
                
                enhancedLog(`Attempting trade with ${pool} pool`, requestBody);

                const response = await fetch(`https://pumpportal.fun/api/trade-local`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(requestBody)
                });

                enhancedLog(`${pool} pool API response`, {
                    status: response.status,
                    headers: Object.fromEntries(response.headers.entries())
                });

                if (response.status === 200) {
                    const data = await response.arrayBuffer();
                    const tx = VersionedTransaction.deserialize(new Uint8Array(data));
                    const signerKeyPair = Keypair.fromSecretKey(bs58.decode(PRIVATE_KEY));
                    tx.sign([signerKeyPair]);
                    const signature = await web3Connection.sendTransaction(tx);
                    
                    return res.json({
                        success: true,
                        signature: signature,
                        link: "https://solscan.io/tx/" + signature,
                        amount: tradeAmount,
                        denominatedInSol: denominatedInSol,
                        pool: pool
                    });
                } else {
                    const errorText = await response.text();
                    enhancedLog(`${pool} pool API error response`, { 
                        status: response.status, 
                        errorText 
                    });
                    
                    // Store the last error in case all pools fail
                    lastError = {
                        status: response.status,
                        errorText,
                        pool
                    };
                }
            } catch (poolError) {
                enhancedLog(`Error trading with ${pool} pool`, {
                    errorMessage: poolError.message,
                    errorStack: poolError.stack
                });
                
                lastError = {
                    status: 500,
                    errorText: poolError.message,
                    pool
                };
            }
        }

        // If we've exhausted all pools, return the last error
        if (lastError) {
            res.status(lastError.status).json({
                success: false,
                error: `Trade failed on all attempted pools. Last error on ${lastError.pool} pool: ${lastError.errorText}`,
                pools: POOLS,
                details: lastError
            });
        }
    } catch (error) {
        enhancedLog('Trade execution error', {
            errorMessage: error.message,
            errorStack: error.stack
        });
        
        res.status(500).json({
            success: false,
            error: error.message,
            stack: error.stack
        });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Trade server running on port ${PORT}`);
});
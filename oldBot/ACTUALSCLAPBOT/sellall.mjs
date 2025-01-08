import { VersionedTransaction, Connection, Keypair, PublicKey } from '@solana/web3.js';
import bs58 from 'bs58';
import { TOKEN_PROGRAM_ID } from '@solana/spl-token';
import fetch from 'node-fetch';

async function sellAllTokens() {
    // Configure Solana RPC endpoint
    const connection = new Connection('https://api.mainnet-beta.solana.com', 'confirmed');

    // Your wallet's public key and private key
    const PRIVATE_KEY = ""
    const PUBLIC_KEY = "6qdXtZFwJWXW7kWSbU6Zphw2DW9xbiCZJY4XqqpKSURM";
    const wallet = Keypair.fromSecretKey(bs58.decode(PRIVATE_KEY));

    try {
        // Get all token accounts for this wallet
        const tokenAccounts = await connection.getParsedTokenAccountsByOwner(
            wallet.publicKey, 
            { programId: TOKEN_PROGRAM_ID }
        );

        // Filter out SOL and WSOL tokens (known mints)
        const NATIVE_SOL_MINT = 'So11111111111111111111111111111111111111112';
        const WRAPPED_SOL_MINT = 'So11111111111111111111111111111111111111112';

        const tokensToSell = tokenAccounts.value
            .filter(account => {
                const mint = account.account.data.parsed.info.mint;
                const balance = account.account.data.parsed.info.tokenAmount.amount;
                return (
                    mint !== NATIVE_SOL_MINT && 
                    mint !== WRAPPED_SOL_MINT && 
                    parseInt(balance) > 0
                );
            })
            .map(account => ({
                mint: account.account.data.parsed.info.mint,
                balance: account.account.data.parsed.info.tokenAmount.amount
            }));

        console.log(`Found ${tokensToSell.length} tokens to sell`);

        // Sell each token via PumpPortal API
        for (const token of tokensToSell) {
            try {
                const requestBody = {
                    "publicKey": PUBLIC_KEY,
                    "action": "sell",
                    "mint": token.mint,
                    "denominatedInSol": "true",
                    "amount": "100%", // Sell entire balance
                    "slippage": 99, // High slippage tolerance
                    "priorityFee": 0.0005,
                    "pool": "pump"
                };

                console.log('Preparing to sell token:', token.mint);

                const response = await fetch(`https://pumpportal.fun/api/trade-local`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(requestBody)
                });

                if (response.status === 200) {
                    const data = await response.arrayBuffer();
                    const tx = VersionedTransaction.deserialize(new Uint8Array(data));
                    const signerKeyPair = Keypair.fromSecretKey(bs58.decode(PRIVATE_KEY));
                    tx.sign([signerKeyPair]);
                    
                    const signature = await connection.sendTransaction(tx);
                    
                    console.log({
                        success: true,
                        signature: signature,
                        link: `https://solscan.io/tx/${signature}`,
                        mint: token.mint
                    });
                } else {
                    console.error(`Failed to sell ${token.mint}. Status: ${response.status}`);
                    const errorText = await response.text();
                    console.error('Error response:', errorText);
                }
            } catch (error) {
                console.error(`Error selling ${token.mint}:`, error);
            }
        }
    } catch (error) {
        console.error('Error finding and selling tokens:', error);
    }
}

// Run the script
sellAllTokens();
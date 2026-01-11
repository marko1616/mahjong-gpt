import multiprocessing as mp
import asyncio
from env import MahjongEnv

# Command Protocol
CMD_RESET = 0
CMD_STEP = 1
CMD_CLOSE = 2

def _worker_loop(pipe, seed: int):
    """
    Independent worker process loop.
    """
    env = MahjongEnv(seed=seed)
    
    try:
        while True:
            cmd, data = pipe.recv()
            
            if cmd == CMD_RESET:
                pipe.send(env.reset())
            elif cmd == CMD_STEP:
                # data is action
                pipe.send(env.step(data))
            elif cmd == CMD_CLOSE:
                break
    except Exception as e:
        print(f"Worker error: {e}")
    finally:
        pipe.close()

class AsyncMahjongEnv:
    """
    Asynchronous proxy for MahjongEnv.
    """
    def __init__(self, seed: int):
        self.ctx = mp.get_context("spawn")
        self.parent_conn, self.child_conn = self.ctx.Pipe()
        self.process = self.ctx.Process(
            target=_worker_loop, 
            args=(self.child_conn, seed)
        )
        self.process.start()
        # We will use the running loop dynamically, so we don't store it here
    
    async def reset(self):
        loop = asyncio.get_running_loop()
        self.parent_conn.send((CMD_RESET, None))
        # Offload the blocking recv to the default executor (thread pool)
        return await loop.run_in_executor(None, self.parent_conn.recv)

    async def step(self, action):
        loop = asyncio.get_running_loop()
        self.parent_conn.send((CMD_STEP, action))
        return await loop.run_in_executor(None, self.parent_conn.recv)

    def close(self):
        self.parent_conn.send((CMD_CLOSE, None))
        self.process.join()


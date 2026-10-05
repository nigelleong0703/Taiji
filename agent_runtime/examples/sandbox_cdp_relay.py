"""Runs inside the E2B sandbox: exposes Chromium's DevTools endpoint (127.0.0.1:9222) on 0.0.0.0:9223.

Chromium answers DevTools HTTP and WebSocket requests only when the Host header is an IP address or
localhost, and refuses WebSocket handshakes that carry an unlisted Origin. Requests arriving through
the E2B proxy name <port>-<id>.<ip>.nip.io, so the relay rewrites Host to localhost:9222 and drops
Origin on each connection's request head, then copies bytes both ways. Standard library only.
"""
import asyncio

LISTEN, CHROME = ("0.0.0.0", 9223), ("127.0.0.1", 9222)


async def pipe(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        writer.close()


async def handle(client_reader, client_writer):
    try:
        head = await client_reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        client_writer.close()
        return
    lines = head.decode("latin-1").split("\r\n")
    kept = [lines[0]] + [line for line in lines[1:] if line and not line.lower().startswith(("host:", "origin:"))]
    kept.insert(1, "Host: localhost:9222")
    chrome_reader, chrome_writer = await asyncio.open_connection(*CHROME)
    chrome_writer.write(("\r\n".join(kept) + "\r\n\r\n").encode("latin-1"))
    await chrome_writer.drain()
    await asyncio.gather(pipe(client_reader, chrome_writer), pipe(chrome_reader, client_writer))


async def main():
    server = await asyncio.start_server(handle, *LISTEN)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())

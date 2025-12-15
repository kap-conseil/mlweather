import asyncio
from httpx_cache import AsyncCacheControlTransport
import httpx

# Module-level variable for lazy initialization
_cache_transport = None


def get_cache_transport():
    """
    Lazily initialize and return the AsyncCacheControlTransport.
    """
    global _cache_transport
    if _cache_transport is None:
        print("Initializing AsyncCacheControlTransport...")
        _cache_transport = AsyncCacheControlTransport()
    return _cache_transport


async def fetch(client, url):
    response = await client.get(url)
    print(f"{url}: {response.status_code}")
    return response.text


async def main(urls):
    # Use the lazily-initialized module-level transport
    async with httpx.AsyncClient(transport=get_cache_transport()) as client:
        tasks = [fetch(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    urls_to_fetch = [
        "https://example.com",
        "https://httpbin.org/get",
        "https://api.github.com",
    ]
    results = asyncio.run(main(urls_to_fetch))

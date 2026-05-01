"""
Docker 컨테이너 리소스 제어
Docker SDK for Python을 통해 컨테이너의 cpu_quota, mem_limit을 동적으로 조정한다.
CLI 호출(docker update) 대신 SDK를 쓰는 이유는 안정성/에러 핸들링이 좋기 때문이다.
"""

import docker

from ai import config


_client = None


def get_client():
    """Docker 클라이언트 싱글톤"""
    global _client
    if _client is None:
        _client = docker.DockerClient(base_url=config.DOCKER_SOCKET)
    return _client


def update_limits(container_name: str, cpu_quota: float, memory_bytes: int) -> dict:
    """
    컨테이너 리소스 limit을 갱신한다.
    적용 전 limit을 반환하여 Watchdog이 롤백 시 사용한다.

    Args:
        container_name: 대상 컨테이너 이름
        cpu_quota: CPU 코어 단위 (예: 0.5)
        memory_bytes: 메모리 limit (bytes)

    Returns:
        이전 설정값 (롤백용)
    """
    # TODO: 구현
    # client = get_client()
    # container = client.containers.get(container_name)
    # prev = {"cpu_quota": container.attrs["HostConfig"]["CpuQuota"],
    #         "memory_bytes": container.attrs["HostConfig"]["Memory"]}
    # container.update(cpu_quota=int(cpu_quota * 100000),
    #                  mem_limit=memory_bytes)
    # return prev
    raise NotImplementedError


def get_current_usage(container_name: str) -> dict:
    """
    현재 컨테이너의 실시간 사용량 (Watchdog용).
    Prometheus를 거치지 않고 Docker에서 직접 읽어 지연을 최소화한다.
    """
    # TODO: 구현
    raise NotImplementedError

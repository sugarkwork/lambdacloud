"""
lambda_cloud_async.py

Lambda Cloud API v1 非同期ラッパー
────────────────────────────────────────────
機能:
  • インスタンス作成 (複数台対応) / 詳細取得 / 一覧取得 / 終了
  • 起動完了待機 (polling)
  • SSHキー一覧取得 + 自動選択
  • 任意ログレベル設定・カスタムロガー差し替え
  • 認証方式選択: "basic" (既定) or "bearer"
依存:
  pip install httpx
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union
import asyncio, time, httpx, logging
from typing import Any, Dict
import os
from dotenv import load_dotenv
load_dotenv()

# ────────────────────────── ロガー ────────────────────────── #

def _default_logger() -> logging.Logger:
    log = logging.getLogger("lambda_cloud")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log

logger = _default_logger()

def set_log_level(level: Union[int, str]) -> None:
    """使用例: set_log_level(logging.DEBUG) あるいは set_log_level('DEBUG')."""
    if isinstance(level, str):
        level = level.upper()
    logger.setLevel(level)

# ────────────────────────── 例外 ────────────────────────── #

class APIError(RuntimeError):
    """HTTP 200 以外や API エラー時に送出"""

# ────────────────── 非同期 Lambda Cloud クライアント ────────────────── #

class AsyncLambdaCloudClient:
    BASE_URL = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self,
                 api_key: str,
                 *,
                 auth_method: str = "basic",   # "basic" or "bearer"
                 timeout: float = 30.0
                 ) -> None:
        self.api_key = api_key
        self.auth_method = auth_method.lower()
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        if self.auth_method not in ("basic", "bearer"):
            raise ValueError("auth_method は 'basic' または 'bearer' を指定してください。")

    async def list_instance_types(self) -> list[InstanceType]:
        """
        起動可能なインスタンスタイプ一覧を取得し、InstanceType オブジェクトのリストで返します。

        Returns:
            List[InstanceType]: name, gpus, vcpus, ram, storage, price_per_hour を含むリスト。
        """
        logger.info("インスタンスタイプ一覧取得リクエスト")
        resp = await self._get("/instance-types")
        items = resp.get("data", {})
        types = [InstanceType.from_json(j) for k,j in items.items()]
        logger.info("インスタンスタイプ %d 件取得", len(types))
        return types

    # ---------- context manager ---------- #
    async def __aenter__(self):
        auth = httpx.BasicAuth(self.api_key, "") if self.auth_method == "basic" else None
        self._client = httpx.AsyncClient(timeout=self._timeout, auth=auth)
        logger.debug("AsyncClient open (auth=%s, timeout=%s)", self.auth_method, self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._client.aclose()
        logger.debug("AsyncClient closed")

    # ---------- 内部 HTTP ---------- #
    def _headers(self) -> dict[str, str]:
        if self.auth_method == "bearer":
            return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        return {"Content-Type": "application/json"}  # Basic 認証時は不要

    async def _get(self, path: str) -> dict:
        assert self._client
        url = self.BASE_URL + path
        logger.debug("GET %s", url)
        res = await self._client.get(url, headers=self._headers())
        logger.debug("→ %s", res.status_code)
        if res.status_code != 200:
            logger.error("GET %s failed: %s", url, res.text)
            raise APIError(res.text)
        return res.json()

    async def _post(self, path: str, json: dict) -> dict:
        assert self._client
        url = self.BASE_URL + path
        logger.debug("POST %s | body=%s", url, json)
        res = await self._client.post(url, headers=self._headers(), json=json)
        logger.debug("→ %s", res.status_code)
        if res.status_code != 200:
            logger.error("POST %s failed: %s", url, res.text)
            raise APIError(res.text)
        return res.json()

    # ---------- SSH Key ---------- #
    async def list_ssh_keys(self) -> List["SSHKey"]:
        keys = [SSHKey.from_json(j) for j in (await self._get("/ssh-keys"))["data"]]
        logger.info("SSH鍵 %d 件取得", len(keys))
        return keys

    # ---------- Instance Factory ---------- #
    async def launch_instance(
        self,
        instance_type_name: str,
        region_name: str,
        *,
        quantity: int = 1,  # 同時に起動するインスタンス数
        ssh_key_name: str | None = None,
        instance_name: str | None = None,
        file_system_name: str | None = None,
    ) -> Union["Instance", List["Instance"]]:
        # SSH Key 自動選択
        if ssh_key_name is None:
            keys = await self.list_ssh_keys()
            if not keys:
                raise RuntimeError("SSH鍵が登録されていません。ダッシュボードから追加してください。")
            if len(keys) == 1:
                ssh_key_name = keys[0].name
                logger.debug("SSH鍵 %s を自動選択", ssh_key_name)
            else:
                raise RuntimeError("SSH鍵が複数あります。ssh_key_name を指定してください。")

        body: dict = {
            "instance_type_name": instance_type_name,
            "region_name": region_name,
            "ssh_key_names": [ssh_key_name]
        }
        if instance_name:
            body["name"] = instance_name
        if file_system_name:
            body["file_system_names"] = [file_system_name]

        logger.info("インスタンス起動要求: type=%s region=%s qty=%d", instance_type_name, region_name, quantity)
        resp = await self._post("/instance-operations/launch", body)
        if 'error' in resp:
            raise Exception(str(resp))
        
        ids = resp.get("data", {}).get("instance_ids", [])  # ID のみ
        # 詳細を取得
        insts = await asyncio.gather(*(self.get_instance(i) for i in ids))
        for inst in insts:
            logger.info("Launched %s status=%s", inst.id, inst.status)
        return insts

    async def get_instance(self, instance_id: str) -> Optional["Instance"]:
        res = await self._get(f"/instances/{instance_id}")
        if not res.get("data"):
            return None
        return Instance.from_json(res["data"], self)

    async def list_instances(self) -> List["Instance"]:
        payload = (await self._get("/instances"))["data"]
        instances = [Instance.from_json(j, self) for j in payload]
        logger.info("稼働中インスタンス %d 件取得", len(instances))
        return instances

    async def _terminate_instance(self, instance_id: str) -> None:
        await self._post("/instance-operations/terminate", {"instance_ids": [instance_id]})
        logger.info("terminate 要求送信 id=%s", instance_id)

# ────────────────────── データモデル ────────────────────── #

@dataclass(slots=True)
class SSHKey:
    id: str
    name: str
    public_key: Optional[str] = None

    @classmethod
    def from_json(cls, j: dict) -> "SSHKey":
        return cls(id=j["id"], name=j["name"], public_key=j.get("public_key"))


@dataclass
class InstanceType:
    name: str
    gpus: int
    vcpus: int
    memory_gib: float         # RAM 容量（GiB）
    storage_gib: float        # ストレージ容量（GiB）
    price_per_hour: float     # 時間あたり料金（ドルなど）
    regions: List[str] = field(default_factory=list)
    raw: Any = field(default=None, repr=False)

    @classmethod
    def from_json(cls, j: dict[str, Any]) -> "InstanceType":
        # "instance_type" の中身を取り出し
        inst = j.get("instance_type", j)
        specs = inst.get("specs", {})

        # 利用可能リージョンをリスト化
        regions = [
            region.get("name", "")
            for region in j.get("regions_with_capacity_available", [])
        ]

        # price_cents_per_hour をドル換算
        price = inst.get("price_cents_per_hour", 0) / 100

        return cls(
            name=inst.get("name", ""),
            gpus=int(specs.get("gpus", 0)),
            vcpus=int(specs.get("vcpus", 0)),
            memory_gib=float(specs.get("memory_gib", 0)),
            storage_gib=float(specs.get("storage_gib", 0)),
            price_per_hour=float(price),
            regions=regions,
            raw=j
        )
    

@dataclass
@dataclass(slots=True)
class Instance:
    id: str
    name: Optional[str]
    status: str
    ip: Optional[str]
    private_ip: Optional[str]
    region: str
    instance_type: InstanceType          # 既存の InstanceType クラスを使いまわせます
    ssh_key_names: List[str]
    file_system_names: List[str]
    hostname: Optional[str]
    actions: Dict[str, Any]
    _client: AsyncLambdaCloudClient = field(repr=False)
    raw: Any = field(default=None, repr=False)

    @classmethod
    def from_json(cls, j: dict[str, Any], client: AsyncLambdaCloudClient) -> Instance:
        # 1) 基本フィールド
        id_ = j["id"]
        name = j.get("name")
        status = j["status"]
        ip = j.get("ip")
        private_ip = j.get("private_ip")

        # 2) region オブジェクト
        region = j["region"]["name"]

        # 3) instance_type はネストされた dict → InstanceType.from_json を使う
        it_j = j["instance_type"]
        instance_type = InstanceType.from_json(it_j)

        # 4) リスト系フィールド
        ssh_key_names = j.get("ssh_key_names", [])
        file_system_names = j.get("file_system_names", [])

        # 5) ホスト名・Jupyter情報
        hostname = j.get("hostname")
        # 6) actions（使うなら中身を解析しても良い）
        actions = j.get("actions", {})

        return cls(
            id=id_,
            name=name,
            status=status,
            ip=ip,
            private_ip=private_ip,
            region=region,
            instance_type=instance_type,
            ssh_key_names=ssh_key_names,
            file_system_names=file_system_names,
            hostname=hostname,
            actions=actions,
            _client=client,
            raw=j
        )
    
    # ----- 操作メソッド ----- #
    async def refresh(self) -> "Instance":
        latest = await self._client.get_instance(self.id)
        if latest is None:
            raise RuntimeError("インスタンスが見つかりません (削除済み?)")
        for field in ("status", "ip", "region", "instance_type", "ssh_key_names", "name"):
            setattr(self, field, getattr(latest, field))
        logger.debug("refresh id=%s status=%s", self.id, self.status)
        return self

    async def wait_until_active(self, timeout: int = 600, interval: int = 20) -> "Instance":
        start = time.time()
        logger.info("起動待機開始 id=%s (timeout=%ds)", self.id, timeout)
        while True:
            await self.refresh()
            if self.status == "active":
                logger.info("起動完了 id=%s ip=%s", self.id, self.ip)
                return self
            if self.status in ("terminating", "terminated"):
                raise RuntimeError(f"インスタンスが {self.status} になりました。")
            if time.time() - start > timeout:
                logger.error("timeout (id=%s, status=%s)", self.id, self.status)
                raise TimeoutError("wait_until_active タイムアウト")
            await asyncio.sleep(interval)

    async def terminate(self) -> None:
        await self._client._terminate_instance(self.id)
        self.status = "terminating"

    # Truthy 判定を “active” に合わせる
    def __bool__(self) -> bool:
        return self.status == "active"


async def launch(gpu=True, price_per_hour=2, name="my-instance-test", quantity=1):
    async with AsyncLambdaCloudClient(os.getenv("LAMBDA_API_KEY")) as client:
        # インスタンスタイプ一覧
        print("------------------------")
        instances = await client.list_instance_types()
        print(instances)

        print("------------------------")
        running_instances = await client.list_instances()
        print(running_instances)

        for i in running_instances:
            if i.name.startswith(name):
                print(i)
                return i

        print("------------------------")
        sshkeys = await client.list_ssh_keys()
        print(sshkeys)

        filter_over = {
            "gpus": 1 if gpu else 0,
            "vcpus": 1,
            "memory_gib": 1,
            "storage_gib": 1,
        }

        filter_under = {
            "price_per_hour": price_per_hour,
        }

        target_instances = []

        for i in instances:
            # region check
            if len(i.regions) == 0:
                continue
            # filter
            if all(getattr(i, k) >= v for k, v in filter_over.items()):
                if any(getattr(i, k) <= v for k, v in filter_under.items()):
                    target_instances.append(i)
        
        target_instances = sorted(target_instances, key=lambda i: i.price_per_hour)

        if len(target_instances) == 0:
            print("フィルタに一致するインスタンスが見つかりませんでした。")
            return

        if len(running_instances) == 0:
            for i in target_instances:
                print(i)

                # 1 インスタンス起動
                insts = await client.launch_instance(
                    i.name, i.regions[0],
                    quantity=quantity,
                    instance_name=f"{name}{i.name}"
                )

                # 複数台戻り値の場合
                await asyncio.gather(*(i.wait_until_active() for i in insts))

                break

        # 2 稼働中一覧
        for i in await client.list_instances():
            i.wait_until_active()
            print(i.id, i.status, i.ip)

        print("インスタンスの終了に成功しました。")
        return
    
    # sudo apt install git-lfs
    # git lfs install
    # git clone https://huggingface.co/spaces/sarulab-speech/CoCoCap-beta
    # cd CoCoCap-beta
    # git lfs pull
    # python biya.py


async def terminate():
    async with AsyncLambdaCloudClient(os.getenv("LAMBDA_API_KEY")) as client:
        running_instances = await client.list_instances()
        # 3 終了
        await asyncio.gather(*(i.terminate() for i in running_instances))
        # 4 インスタンス削除
        await asyncio.gather(*(i.terminate() for i in running_instances))


async def main():
    set_log_level(logging.DEBUG)  # 詳細ログ
    instance = await launch(gpu=True, price_per_hour=2, name="my-instance-test")

    print(instance)

    await asyncio.sleep(10)

    #await terminate()

    await instance.terminate()



if __name__ == "__main__":
    asyncio.run(main())

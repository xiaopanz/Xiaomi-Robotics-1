# Copyright (C) 2026 Xiaomi Corporation.
import pickle
import socket
import struct
import time
import argparse
import traceback

import torch
from tqdm import tqdm
from transformers import AutoModel
import trace_rpc


class Server:
    def __init__(self, model_path, host, port):
        self.host = host
        self.port = port
        self.model_path = model_path
        self._trace_identity = None

        print("Loading model...")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16).cuda().to(torch.bfloat16)
        print("Model loaded.")

    def handle(self, request):
        op = request.get('_trace_op')
        if op is not None:
            if request.get('protocol') != trace_rpc.PROTOCOL:
                raise ValueError('Unsupported TRACE-MM protocol')
            if op == 'hello':
                if self._trace_identity is None:
                    self._trace_identity = trace_rpc.model_identity(self.model_path)
                return {'protocol': trace_rpc.PROTOCOL, 'model': self._trace_identity,
                        'runtime': trace_rpc.runtime_identity(), 'training': self.model.training}
            if op == 'get_rng':
                return trace_rpc.get_rng()
            if op == 'set_rng':
                trace_rpc.set_rng(request['state'])
                return {'ok': True}
            if op == 'seed':
                trace_rpc.seed_rng(request['seed'])
                return trace_rpc.get_rng()
            if op != 'infer':
                raise ValueError(f'Unknown TRACE-MM operation: {op}')
            request = request['request']
            before = trace_rpc.get_rng()
            fingerprint = trace_rpc.request_hash(request)
        data = {key: (value.to(device=self.model.device, dtype=self.model.dtype)
                      if isinstance(value, torch.Tensor) and value.is_floating_point()
                      else value.to(device=self.model.device) if isinstance(value, torch.Tensor)
                      else value) for key, value in request.items()}
        outputs = self.model(**data)
        actions = outputs.actions.cpu()
        if op == 'infer':
            return {'actions': actions, 'rng_before': before, 'rng_after': trace_rpc.get_rng(),
                    'request_sha256': fingerprint}
        return actions

    def _recv_all(self, conn, length):
        data = b""
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"Server running on {self.host}:{self.port}...")

            while True:
                conn, addr = server_socket.accept()

                try:
                    request_count = 0
                    with tqdm(desc="Processing Requests", unit=" req") as pbar:
                        while True:
                            data_len_bytes = self._recv_all(conn, 4)
                            if not data_len_bytes:
                                break
                            data_len = struct.unpack(">I", data_len_bytes)[0]

                            data = self._recv_all(conn, data_len)
                            if not data:
                                break

                            tic = time.time()

                            input_data = pickle.loads(data)
                            response = pickle.dumps(self.handle(input_data))
                            conn.sendall(struct.pack(">I", len(response)) + response)

                            toc = time.time()
                            request_count += 1
                            pbar.update(1)
                            pbar.set_postfix(
                                {
                                    "avg_time": f"{(toc - tic) * 1000:.2f}ms",
                                }
                            )

                except Exception as e:
                    print(f"Error handling connection: {e}")
                    traceback.print_exc()
                finally:
                    conn.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model dir.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10086,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    server = Server(model_path=args.model, host=args.host, port=args.port)
    server.serve()

"""Pytest conftest — isola a suíte das dependências pesadas de GPU.

Carregar `torch` de verdade sob o pytest no Windows provocava crash fatal (access
violation) durante o load das DLLs. A versão anterior resolvia isso injetando um
MagicMock em `sys.modules["torch"]`, mas isso quebrava em qualquer ambiente com
`ultralytics` instalado: o ultralytics lê `torch.__version__` e `torch.jit.TracerWarning`
no momento do import, e um MagicMock não satisfaz nem dunders nem `issubclass`. O
resultado era a coleta do pytest abortando inteira — a suíte só passava em interpretadores
onde o ultralytics não estava instalado (por isso nunca rodou dentro do venv do projeto).

A abordagem aqui é bloquear o import de verdade, fazendo-o levantar ImportError. Todos os
consumidores desses módulos no projeto já tratam ImportError e degradam graciosamente
(`object_detector.YOLO_AVAILABLE`, `runtime_config.detect_vram_gb`, etc.), então a suíte
exercita o mesmo caminho de fallback que roda numa máquina sem GPU — sem mock nenhum.

Testes que precisem de torch de verdade devem usar `pytest.importorskip("torch")`, que
trata o ImportError e marca o teste como skipped.
"""

import sys

# Módulos bloqueados: pesados, dependentes de DLL/CUDA e desnecessários para a suíte.
BLOCKED_MODULES = frozenset({
    "torch",
    "torchvision",
    "torchaudio",
    "ultralytics",
    "detectron2",
})


class _BlockedImportFinder:
    """Meta path finder que recusa os módulos pesados com ImportError."""

    def find_module(self, fullname, path=None):  # protocolo legado, mantido por segurança
        return self if self._is_blocked(fullname) else None

    def find_spec(self, fullname, path=None, target=None):
        if self._is_blocked(fullname):
            raise ImportError(
                f"'{fullname}' está bloqueado durante os testes (veja tests/conftest.py). "
                f"Use pytest.importorskip se o teste realmente precisar dele."
            )
        return None

    def load_module(self, fullname):
        raise ImportError(f"'{fullname}' está bloqueado durante os testes")

    @staticmethod
    def _is_blocked(fullname):
        raiz = fullname.split(".")[0]
        return raiz in BLOCKED_MODULES


# Remove o que porventura já tenha sido importado antes do conftest e instala o bloqueio
# na frente da cadeia de importação.
for _nome in list(sys.modules):
    if _nome.split(".")[0] in BLOCKED_MODULES:
        del sys.modules[_nome]

if not any(isinstance(f, _BlockedImportFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockedImportFinder())

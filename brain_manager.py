"""Brain Weight Manager: Reset to baseline, rollback to previous sessions, and checkpoint archive management.

This module provides tools for managing the Drosophila plastic brain weights:
- Reset to clean biological baseline weights (unlearned physiological initial state).
- Rollback to the previous training session checkpoint.
- Inspect current model parameters, fitness, generation, and historical checkpoints.
- Restore specific historical checkpoints from archive.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

import config
from fly_brain import FlyBrain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BrainManager")

# Reconfigure stdout/stderr to UTF-8 on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class BrainManager:
    """Manages FlyBrain weight lifecycle: baselines, rollbacks, and checkpoint archives."""

    def __init__(
        self,
        active_weights_path: Path = config.BEST_BRAIN_WEIGHTS_FILE,
        previous_weights_path: Path = config.PREVIOUS_BRAIN_WEIGHTS_FILE,
        baseline_weights_path: Path = config.BASELINE_BRAIN_WEIGHTS_FILE,
        checkpoints_dir: Path = config.CHECKPOINTS_DIR,
    ) -> None:
        self.active_path = active_weights_path
        self.previous_path = previous_weights_path
        self.baseline_path = baseline_weights_path
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def backup_current(self, tag: str = "manual", update_previous: bool = True) -> Optional[Path]:
        """Create a backup of the current active weights file."""
        if not self.active_path.exists():
            return None

        # 1. Update immediate rollback pointer if requested
        if update_previous:
            shutil.copy2(self.active_path, self.previous_path)

        # 2. Archive to historical folder
        ckpt_meta = self.get_checkpoint_info(self.active_path)
        gen = ckpt_meta.get("generation", 0)
        fit = ckpt_meta.get("best_fitness", 0.0)
        fit_str = f"{fit:.1f}".replace("-", "neg")
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        archive_name = f"brain_weights_{timestamp_str}_gen{gen}_fit{fit_str}_{tag}.pt"
        archive_path = self.checkpoints_dir / archive_name
        shutil.copy2(self.active_path, archive_path)
        return archive_path

    def reset_to_baseline(self) -> bool:
        """Reset active weights to unlearned, biological baseline weights."""
        print("\n" + "=" * 65)
        print(" СБРОС МОЗГА ДО ИСХОДНОГО СОСТОЯНИЯ (BIOLOGICAL BASELINE RESET)")
        print("=" * 65)

        # Safety backup of current weights
        if self.active_path.exists():
            backup_path = self.backup_current(tag="pre_reset_backup")
            if backup_path:
                print(f" [*] Текущие веса сохранены в резервную копию:\n     {backup_path.name}")

        brain = FlyBrain()
        brain.reset_to_baseline()

        # Save immutable reference baseline if not present
        if not self.baseline_path.exists():
            brain.save_baseline(self.baseline_path)

        # Save to active weights path
        metadata = {
            "type": "biological_baseline",
            "best_fitness": 0.0,
            "generation": 0,
            "candidate_idx": 0,
            "saved_at": time.time(),
            "description": "Default physiological weights (Hebbian receptive fields + descending motor thresholds)",
        }
        brain.save_weights(self.active_path, metadata=metadata)

        print(f" [OK] Мозг успешно сброшен до исходного состояния!")
        print(f" Файл активных весов: {self.active_path}")
        print(" Параметры: Поколение = 0, Фитнес = 0.0, Пластические слои возвращены к биологическому дефолту.")
        print("=" * 65 + "\n")
        return True

    def rollback_to_previous(self) -> bool:
        """Rollback active weights to the previous session checkpoint."""
        print("\n" + "=" * 65)
        print(" ОТКАТ МОЗГА ДО РЕЗУЛЬТАТОВ ПРЕДЫДУЩЕЙ ТРЕНИРОВКИ (ROLLBACK)")
        print("=" * 65)

        source_path: Optional[Path] = None

        # Check immediate previous session file
        if self.previous_path.exists():
            source_path = self.previous_path
        else:
            # Fallback to the latest checkpoint in checkpoints directory
            archives = self.list_archived_checkpoints()
            if archives:
                source_path = archives[0]["path"]

        if not source_path or not source_path.exists():
            print(" [ОШИБКА] Резервная копия прошлой тренировки не найдена!")
            print(f" Ожидался файл: {self.previous_path}")
            print(f" Или архивные копии в: {self.checkpoints_dir}")
            print("=" * 65 + "\n")
            return False

        # Create temporary safety backup of active before overwriting (do not overwrite previous session pointer)
        if self.active_path.exists():
            self.backup_current(tag="before_rollback", update_previous=False)

        # Copy previous to active
        shutil.copy2(source_path, self.active_path)
        info = self.get_checkpoint_info(self.active_path)

        print(f" [OK] Успешный откат до контрольной точки прошлой сессии!")
        print(f" Источник восстановления: {source_path.name}")
        print(f" Номер поколения:        {info.get('generation', 'N/A')}")
        print(f" Достигнутый фитнес:     {info.get('best_fitness', 'N/A')}")
        if "saved_at" in info:
            dt = datetime.datetime.fromtimestamp(info["saved_at"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f" Время сохранения:       {dt}")
        print("=" * 65 + "\n")
        return True

    def restore_checkpoint(self, checkpoint_path: Path) -> bool:
        """Restore a specific checkpoint file to active weights."""
        if not checkpoint_path.exists():
            print(f"[ОШИБКА] Файл чекпоинта не найден: {checkpoint_path}")
            return False

        if self.active_path.exists():
            self.backup_current(tag="before_restore")

        shutil.copy2(checkpoint_path, self.active_path)
        info = self.get_checkpoint_info(self.active_path)
        print(f"\n[OK] Чекпоинт {checkpoint_path.name} успешно установлен как активный!")
        print(f"  Поколение: {info.get('generation', 'N/A')} | Фитнес: {info.get('best_fitness', 'N/A')}\n")
        return True

    def get_checkpoint_info(self, path: Path) -> Dict[str, Any]:
        """Extract metadata dictionary from a PyTorch .pt checkpoint."""
        if not path.exists():
            return {}
        try:
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict):
                return {
                    "best_fitness": ckpt.get("best_fitness", 0.0),
                    "generation": ckpt.get("generation", 0),
                    "candidate_idx": ckpt.get("candidate_idx", 0),
                    "saved_at": ckpt.get("saved_at", path.stat().st_mtime),
                    "type": ckpt.get("type", "trained"),
                    "description": ckpt.get("description", ""),
                }
        except Exception as e:
            logger.debug(f"Could not read metadata from {path}: {e}")
        return {"saved_at": path.stat().st_mtime}

    def list_archived_checkpoints(self) -> List[Dict[str, Any]]:
        """List all historical checkpoints sorted by modification time (newest first)."""
        checkpoints = []
        if not self.checkpoints_dir.exists():
            return []

        for p in self.checkpoints_dir.glob("*.pt"):
            info = self.get_checkpoint_info(p)
            info["path"] = p
            info["filename"] = p.name
            info["size_kb"] = round(p.stat().st_size / 1024.0, 1)
            checkpoints.append(info)

        checkpoints.sort(key=lambda x: x.get("saved_at", 0), reverse=True)
        return checkpoints

    def print_status(self) -> None:
        """Display comprehensive brain status and checkpoint availability."""
        print("\n" + "=" * 68)
        print("          СТАТУС И СОСТОЯНИЕ МОЗГА ДРОЗОФИЛЫ (POLYTRACK)")
        print("=" * 68)

        # 1. Active Weights
        print("\n[1] АКТИВНАЯ МОДЕЛЬ (data/brain_weights_best.pt):")
        if self.active_path.exists():
            info = self.get_checkpoint_info(self.active_path)
            dt = datetime.datetime.fromtimestamp(info.get("saved_at", 0)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"    - Статус:         ДОСТУПНА ({self.active_path.stat().st_size / 1024:.1f} KB)")
            print(f"    - Тип модели:     {info.get('type', 'Обученная модель')}")
            print(f"    - Лучший фитнес:  {info.get('best_fitness', 0.0):.2f}")
            print(f"    - Поколение (Gen): {info.get('generation', 0)}")
            print(f"    - Дата записи:    {dt}")
            if info.get("description"):
                print(f"    - Описание:       {info['description']}")
        else:
            print("    - Статус:         ОТСУТСТВУЕТ (мозг работает на исходных дефолтах)")

        # 2. Previous Session Rollback Point
        print("\n[2] ТОЧКА ОТКАТА (data/brain_weights_previous.pt):")
        if self.previous_path.exists():
            p_info = self.get_checkpoint_info(self.previous_path)
            dt_p = datetime.datetime.fromtimestamp(p_info.get("saved_at", 0)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"    - Статус:         ДОСТУПНА ДЛЯ ОТКАТА")
            print(f"    - Фитнес:         {p_info.get('best_fitness', 0.0):.2f}")
            print(f"    - Поколение:      {p_info.get('generation', 0)}")
            print(f"    - Дата:           {dt_p}")
        else:
            print("    - Статус:         НЕТ РЕЗЕРВНОЙ КОПИИ ПРОШЛОЙ СЕССИИ")

        # 3. Historical Archives
        archives = self.list_archived_checkpoints()
        print(f"\n[3] АРХИВ ЧЕКПОИНТОВ ({len(archives)} сохраненных копий в data/checkpoints/):")
        if archives:
            for i, a in enumerate(archives[:8]):
                dt_a = datetime.datetime.fromtimestamp(a.get("saved_at", 0)).strftime("%m-%d %H:%M")
                fit_val = a.get("best_fitness", 0.0)
                gen_val = a.get("generation", 0)
                print(f"    [{i + 1}] {a['filename']:42s} | Gen: {gen_val:2d} | Fit: {fit_val:5.1f} | {dt_a}")
            if len(archives) > 8:
                print(f"    ... и еще {len(archives) - 8} архивных файлов.")
        else:
            print("    - Архив пуст (резервные копии создаются автоматически при обучении).")

        print("\n" + "=" * 68 + "\n")


def interactive_menu(manager: BrainManager) -> None:
    """Run interactive text menu for Windows batch runner."""
    while True:
        print("===================================================================")
        print("      УПРАВЛЕНИЕ МОЗГОМ ДРОЗОФИЛЫ (Polytrack Brain Manager)")
        print("===================================================================")
        print("  [1] Сбросить мозг до исходного биологического состояния (Baseline)")
        print("  [2] Откатить мозг до результатов прошлой тренировки (Rollback)")
        print("  [3] Показать текущий статус и информацию о весах (Status)")
        print("  [4] Восстановить выбранный чекпоинт из архива (Restore Archive)")
        print("  [0] Выход")
        print("-------------------------------------------------------------------")

        choice = input("Выберите действие [0-4]: ").strip()
        if choice == "1":
            confirm = input("Вы уверены, что хотите сбросить мозг до исходного состояния? [y/N]: ").strip().lower()
            if confirm in ["y", "yes", "д", "да"]:
                manager.reset_to_baseline()
            else:
                print("Сброс отменен.")
        elif choice == "2":
            manager.rollback_to_previous()
        elif choice == "3":
            manager.print_status()
        elif choice == "4":
            archives = manager.list_archived_checkpoints()
            if not archives:
                print("\n[!] В архиве нет сохраненных чекпоинтов.\n")
                continue
            print("\nДоступные контрольные точки:")
            for i, a in enumerate(archives):
                print(f"  [{i + 1}] {a['filename']} (Gen: {a.get('generation', 0)}, Fit: {a.get('best_fitness', 0.0):.1f})")
            idx_str = input("\nВведите номер чекпоинта для восстановления: ").strip()
            if idx_str.isdigit() and 1 <= int(idx_str) <= len(archives):
                manager.restore_checkpoint(archives[int(idx_str) - 1]["path"])
            else:
                print("Некорректный выбор.")
        elif choice in ["0", "q", "exit"]:
            print("Выход.")
            break
        else:
            print("Неверный ввод, попробуйте еще раз.")
        print()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Drosophila Brain Weight Manager (Module 5)")
    parser.add_argument("--reset-baseline", action="store_true", help="Reset active weights to biological baseline")
    parser.add_argument("--rollback", action="store_true", help="Rollback active weights to previous session")
    parser.add_argument("--status", action="store_true", help="Display brain status and checkpoint list")
    parser.add_argument("--list", action="store_true", help="List all historical checkpoints")
    parser.add_argument("--restore", type=str, default=None, help="Restore specific checkpoint by path or filename")

    args = parser.parse_args()
    manager = BrainManager()

    if args.reset_baseline:
        manager.reset_to_baseline()
        return

    if args.rollback:
        manager.rollback_to_previous()
        return

    if args.status:
        manager.print_status()
        return

    if args.list:
        archives = manager.list_archived_checkpoints()
        print(f"\nАрхивных чекпоинтов: {len(archives)}")
        for a in archives:
            print(f"  {a['filename']} | Gen: {a.get('generation', 0)} | Fit: {a.get('best_fitness', 0.0):.1f}")
        return

    if args.restore:
        p = Path(args.restore)
        if not p.is_absolute():
            # Check in checkpoints dir
            cand = config.CHECKPOINTS_DIR / p.name
            if cand.exists():
                p = cand
        manager.restore_checkpoint(p)
        return

    # If no flags passed, open interactive menu
    interactive_menu(manager)


if __name__ == "__main__":
    main()

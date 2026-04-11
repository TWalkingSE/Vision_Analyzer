#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Object Detector - Detecção de Objetos com YOLO11 e Detectron2
=================================================================
Detecta objetos usando YOLO11 (padrão) ou Detectron2 (cenas complexas).
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from io import BytesIO

logger = logging.getLogger(__name__)

# Verificar dependências
try:
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("⚠️ ultralytics não instalado. Instale com: pip install ultralytics")

# Detectron2
DETECTRON2_AVAILABLE = False
try:
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    pass


@dataclass
class Detection:
    """Uma detecção de objeto."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]


@dataclass
class DetectionResult:
    """Resultado completo da detecção."""
    image_path: str
    model_name: str
    detections: List[Detection] = field(default_factory=list)
    total_objects: int = 0
    processing_time: float = 0.0
    image_size: Tuple[int, int] = (0, 0)
    
    def get_by_class(self, class_name: str) -> List[Detection]:
        """Retorna detecções de uma classe específica."""
        return [d for d in self.detections if d.class_name.lower() == class_name.lower()]
    
    def get_summary(self) -> Dict[str, int]:
        """Retorna contagem por classe."""
        summary = {}
        for d in self.detections:
            summary[d.class_name] = summary.get(d.class_name, 0) + 1
        return summary


# Classes de interesse para análise forense
FORENSIC_CLASSES = {
    "person": "👤 Pessoa",
    "car": "🚗 Carro",
    "truck": "🚚 Caminhão",
    "motorcycle": "🏍️ Moto",
    "bicycle": "🚲 Bicicleta",
    "bus": "🚌 Ônibus",
    "cell phone": "📱 Celular",
    "knife": "🔪 Faca",
    "scissors": "✂️ Tesoura",
    "backpack": "🎒 Mochila",
    "handbag": "👜 Bolsa",
    "suitcase": "🧳 Mala",
    "bottle": "🍾 Garrafa",
    "laptop": "💻 Notebook",
    "tv": "📺 TV/Monitor",
    "dog": "🐕 Cachorro",
    "cat": "🐈 Gato",
}


class ObjectDetector:
    """Detector de objetos usando YOLO11."""
    
    AVAILABLE_MODELS = {
        "yolo11n": "YOLO11-N (Nano/Super Rápido)",
        "yolo11s": "YOLO11-S (Pequeno/Rápido)",
        "yolo11m": "YOLO11-M (Médio/Balanceado)",
        "yolo11l": "YOLO11-L (Grande/Preciso)",
        "yolo11x": "YOLO11-X (Extra Grande/Máxima Precisão)",
    }
    
    def __init__(self, model_name: str = "yolo11m", confidence_threshold: float = 0.25):
        """
        Inicializa o detector.
        
        Args:
            model_name: Nome do modelo (yolo11s, yolo11m, yolo11l)
            confidence_threshold: Limiar de confiança (0-1)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._loaded = False
        
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics não instalado")
    
    def load_model(self):
        """Carrega o modelo (lazy loading)."""
        if self._loaded:
            return
        
        try:
            logger.info(f"🔄 Carregando modelo {self.model_name}...")
            self.model = YOLO(f"{self.model_name}.pt")
            self._loaded = True
            logger.info(f"✅ Modelo {self.model_name} carregado")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise
    
    def detect(
        self, 
        image: Image.Image,
        classes_filter: List[str] = None
    ) -> DetectionResult:
        """
        Detecta objetos na imagem.
        
        Args:
            image: Imagem PIL
            classes_filter: Lista de classes para filtrar (None = todas)
            
        Returns:
            DetectionResult com todas as detecções
        """
        import time
        start_time = time.time()
        
        if not self._loaded:
            self.load_model()
        
        # Converter para numpy
        img_array = np.array(image)
        
        # Executar detecção
        results = self.model(img_array, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None:
                continue
            
            for i, box in enumerate(boxes):
                # Extrair informações
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                # Filtrar classes se especificado
                if classes_filter and cls_name.lower() not in [c.lower() for c in classes_filter]:
                    continue
                
                # Calcular centro
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                detection = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(center_x, center_y)
                )
                detections.append(detection)
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            image_path="",
            model_name=self.model_name,
            detections=detections,
            total_objects=len(detections),
            processing_time=processing_time,
            image_size=image.size
        )
    
    def detect_and_draw(
        self, 
        image: Image.Image,
        classes_filter: List[str] = None,
        draw_labels: bool = True,
        draw_confidence: bool = True
    ) -> Tuple[Image.Image, DetectionResult]:
        """
        Detecta objetos e desenha bounding boxes na imagem.
        
        Returns:
            Tuple[imagem_com_boxes, resultado_detecção]
        """
        from PIL import ImageDraw, ImageFont
        
        result = self.detect(image, classes_filter)
        
        # Criar cópia para desenhar
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Tentar carregar fonte
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Cores por classe
        colors = {}
        base_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]
        
        for det in result.detections:
            # Cor para a classe
            if det.class_name not in colors:
                colors[det.class_name] = base_colors[len(colors) % len(base_colors)]
            color = colors[det.class_name]
            
            # Desenhar box
            x1, y1, x2, y2 = det.bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Desenhar label
            if draw_labels:
                label = det.class_name
                if draw_confidence:
                    label += f" {det.confidence:.0%}"
                
                # Background do texto
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
                draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
        
        return img_draw, result
    
    def extract_crops(
        self,
        image: Image.Image,
        result: DetectionResult = None,
        padding: int = 10
    ) -> List[Tuple[Image.Image, Detection]]:
        """
        Extrai sub-imagens (crops) de cada objeto detectado.
        
        Args:
            image: Imagem PIL original
            result: DetectionResult pré-existente (ou None para rodar detect)
            padding: Pixels de margem ao redor da bbox
            
        Returns:
            Lista de (crop_image, detection)
        """
        if result is None:
            result = self.detect(image)
        
        crops = []
        w, h = image.size
        
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            # Adicionar padding respeitando limites da imagem
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = image.crop((x1, y1, x2, y2))
            crops.append((crop, det))
        
        return crops

    def get_forensic_summary(self, result: DetectionResult) -> str:
        """Gera resumo forense das detecções."""
        lines = ["## 🎯 Detecção de Objetos (YOLO)", ""]
        lines.append(f"**Modelo:** {result.model_name}")
        lines.append(f"**Tempo:** {result.processing_time:.2f}s")
        lines.append(f"**Total de objetos:** {result.total_objects}")
        lines.append("")
        
        if result.total_objects == 0:
            lines.append("*Nenhum objeto detectado*")
            return "\n".join(lines)
        
        lines.append("### Objetos Detectados")
        lines.append("")
        lines.append("| Objeto | Quantidade | Confiança Média |")
        lines.append("|--------|------------|-----------------|")
        
        summary = result.get_summary()
        for cls_name, count in sorted(summary.items(), key=lambda x: -x[1]):
            # Calcular confiança média
            dets = result.get_by_class(cls_name)
            avg_conf = sum(d.confidence for d in dets) / len(dets)
            
            # Ícone se disponível
            icon = FORENSIC_CLASSES.get(cls_name.lower(), "📦")
            if icon.startswith("📦"):
                icon = f"📦 {cls_name}"
            
            lines.append(f"| {icon} | {count} | {avg_conf:.0%} |")
        
        return "\n".join(lines)


# Instância singleton
_detector_instances: Dict[str, ObjectDetector] = {}


def get_detector(model_name: str = "yolo11m") -> Optional[ObjectDetector]:
    """Retorna instância singleton do detector."""
    if not YOLO_AVAILABLE:
        return None
    
    if model_name not in _detector_instances:
        try:
            _detector_instances[model_name] = ObjectDetector(model_name)
        except Exception as e:
            logger.error(f"Erro ao criar detector: {e}")
            return None
    
    return _detector_instances[model_name]


def is_yolo_available() -> bool:
    """Verifica se YOLO está disponível."""
    return YOLO_AVAILABLE


def is_detectron2_available() -> bool:
    """Verifica se Detectron2 está disponível."""
    return DETECTRON2_AVAILABLE


class Detectron2Detector:
    """Detector de objetos usando Detectron2 (para cenas complexas)."""
    
    AVAILABLE_MODELS = {
        "faster_rcnn_R_50_FPN": "Faster R-CNN R50-FPN (Balanceado)",
        "faster_rcnn_R_101_FPN": "Faster R-CNN R101-FPN (Preciso)",
        "mask_rcnn_R_50_FPN": "Mask R-CNN R50-FPN (Segmentação)",
    }
    
    MODEL_ZOO_CONFIGS = {
        "faster_rcnn_R_50_FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "faster_rcnn_R_101_FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        "mask_rcnn_R_50_FPN": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    }
    
    def __init__(self, model_name: str = "faster_rcnn_R_50_FPN", confidence_threshold: float = 0.5):
        if not DETECTRON2_AVAILABLE:
            raise RuntimeError("detectron2 não instalado")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.predictor = None
        self.metadata = None
        self._loaded = False
    
    def load_model(self):
        """Carrega o modelo Detectron2."""
        if self._loaded:
            return
        
        config_path = self.MODEL_ZOO_CONFIGS.get(self.model_name)
        if not config_path:
            raise ValueError(f"Modelo desconhecido: {self.model_name}")
        
        logger.info(f"🔄 Carregando Detectron2 {self.model_name}...")
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(cfg)
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self._loaded = True
        logger.info(f"✅ Detectron2 {self.model_name} carregado")
    
    def detect(self, image: Image.Image, classes_filter: List[str] = None) -> DetectionResult:
        """Detecta objetos usando Detectron2."""
        import time
        start_time = time.time()
        
        if not self._loaded:
            self.load_model()
        
        # Converter PIL para numpy BGR (Detectron2 espera BGR)
        img_array = np.array(image.convert('RGB'))
        img_bgr = img_array[:, :, ::-1]
        
        outputs = self.predictor(img_bgr)
        instances = outputs["instances"].to("cpu")
        
        detections = []
        class_names = self.metadata.thing_classes
        
        for i in range(len(instances)):
            box = instances.pred_boxes[i].tensor[0].numpy()
            cls_id = int(instances.pred_classes[i])
            conf = float(instances.scores[i])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            
            if classes_filter and cls_name.lower() not in [c.lower() for c in classes_filter]:
                continue
            
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                center=(center_x, center_y)
            ))
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            image_path="",
            model_name=f"detectron2_{self.model_name}",
            detections=detections,
            total_objects=len(detections),
            processing_time=processing_time,
            image_size=image.size
        )
    
    def get_forensic_summary(self, result: DetectionResult) -> str:
        """Gera resumo forense das detecções Detectron2."""
        lines = ["## 🎯 Detecção de Objetos (Detectron2)", ""]
        lines.append(f"**Modelo:** {result.model_name}")
        lines.append(f"**Tempo:** {result.processing_time:.2f}s")
        lines.append(f"**Total de objetos:** {result.total_objects}")
        lines.append("")
        
        if result.total_objects == 0:
            lines.append("*Nenhum objeto detectado*")
            return "\n".join(lines)
        
        lines.append("### Objetos Detectados")
        lines.append("")
        lines.append("| Objeto | Quantidade | Confiança Média |")
        lines.append("|--------|------------|-----------------|")
        
        summary = result.get_summary()
        for cls_name, count in sorted(summary.items(), key=lambda x: -x[1]):
            dets = result.get_by_class(cls_name)
            avg_conf = sum(d.confidence for d in dets) / len(dets)
            icon = FORENSIC_CLASSES.get(cls_name.lower(), f"📦 {cls_name}")
            lines.append(f"| {icon} | {count} | {avg_conf:.0%} |")
        
        return "\n".join(lines)


# Detectron2 singleton
_detectron2_instances: Dict[str, Detectron2Detector] = {}


def get_detectron2_detector(model_name: str = "faster_rcnn_R_50_FPN") -> Optional[Detectron2Detector]:
    """Retorna instância singleton do detector Detectron2."""
    if not DETECTRON2_AVAILABLE:
        return None
    
    if model_name not in _detectron2_instances:
        try:
            _detectron2_instances[model_name] = Detectron2Detector(model_name)
        except Exception as e:
            logger.error(f"Erro ao criar Detectron2 detector: {e}")
            return None
    
    return _detectron2_instances[model_name]

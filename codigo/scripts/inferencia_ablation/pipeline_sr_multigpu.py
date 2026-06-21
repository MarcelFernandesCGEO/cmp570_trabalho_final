"""
Pipeline de Super-Resolução Multi-GPU para servidor Linux.
Processa múltiplos rasters em paralelo, um por GPU.

Fluxo por raster:
  GeoTIFF → numpy (RAM) → tiles → GPU (batches) → mosaico → GeoTIFF

Uso:
  # Processar um raster (usa GPU 0 por padrão)
  python pipeline_sr_multigpu.py -opt config_server.yml \
    -input /data/rasters/t22jbn_A.tif \
    -output /data/output/t22jbn_A_super.tif \
    -imagens 0 1 2 3 -ponderacao linear

  # Processar múltiplos rasters em paralelo (1 por GPU)
  python pipeline_sr_multigpu.py -opt config_server.yml \
    -input /data/rasters/tile_A.tif /data/rasters/tile_B.tif /data/rasters/tile_C.tif \
    -output /data/output/tile_A_sr.tif /data/output/tile_B_sr.tif /data/output/tile_C_sr.tif \
    -imagens 0 1 2 3 -ponderacao linear

  # Processar pasta inteira (distribui automaticamente entre GPUs)
  python pipeline_sr_multigpu.py -opt config_server.yml \
    -input_dir /data/rasters/ \
    -output_dir /data/output/ \
    -imagens 0 1 2 3 -ponderacao linear
"""

import os
import argparse
import numpy as np
import time
import torch
import random
import glob
from multiprocessing import Process, Queue

from osgeo import gdal

from ssr.utils.options import yaml_load
from ssr.utils.model_utils import build_network


def criar_mascara_ponderacao(altura, largura, tipo='linear'):
    """Cria máscara de ponderação para fusão de tiles sobrepostos."""
    y = np.linspace(0, 1, altura)
    x = np.linspace(0, 1, largura)
    xx, yy = np.meshgrid(x, y)
    dist_x = np.minimum(xx, 1 - xx) * 2
    dist_y = np.minimum(yy, 1 - yy) * 2
    dist = np.minimum(dist_x, dist_y)

    if tipo == 'gaussiana':
        mascara = np.exp(-(1 - dist)**2 / (2 * 0.3**2))
    elif tipo == 'coseno':
        mascara = 0.5 * (1 - np.cos(np.pi * dist))
    else:
        mascara = dist

    return np.clip(mascara, 0, 1).astype(np.float32)


def prepare_tensor_from_tile(tile_data, n_lr_images, device):
    """
    Converte tile numpy (imagens_stack*3, H, W) → tensor GPU (1, n_lr_images*3, H, W).
    """
    n_channels = tile_data.shape[0]
    h, w = tile_data.shape[1], tile_data.shape[2]

    n_imgs = n_channels // 3
    chunks = tile_data.reshape(n_imgs, 3, h, w)

    goods, bads = [], []
    for i in range(n_imgs):
        img_hwc = chunks[i].transpose(1, 2, 0)
        if np.any(np.all(img_hwc == 0, axis=2)):
            bads.append(i)
        else:
            goods.append(i)

    if len(goods) >= n_lr_images:
        indices = random.sample(goods, n_lr_images)
    else:
        need = n_lr_images - len(goods)
        indices = goods + random.sample(bads, min(need, len(bads)))
        while len(indices) < n_lr_images:
            indices.append(random.choice(goods if goods else bads))

    selected = chunks[indices]
    tensor = torch.from_numpy(selected.reshape(1, n_lr_images * 3, h, w)).float().to(device) / 255.0
    return tensor


def process_single_raster(gpu_id, raster_path, output_path, opt_path,
                          imagens_selecionadas, tamanho_tile, sobreposicao,
                          batch_size, tipo_ponderacao, result_queue):
    """
    Processa um único raster numa GPU específica.
    Roda como processo separado para multi-GPU.
    """
    tag = f"[GPU {gpu_id}]"
    t_total = time.time()

    try:
        device = torch.device(f'cuda:{gpu_id}')
        opt = yaml_load(opt_path)
        n_lr_images = opt['n_lr_images']

        # Carregar modelo nesta GPU
        print(f"{tag} Carregando modelo...", flush=True)
        model = build_network(opt)
        if 'pretrain_network_g' in opt['path']:
            weights = opt['path']['pretrain_network_g']
            state_dict = torch.load(weights, map_location=device)
            model.load_state_dict(state_dict[opt['path']['param_key_g']],
                                  strict=opt['path']['strict_load_g'])
        model = model.to(device).eval()

        # Ler raster
        raster_name = os.path.basename(raster_path)
        print(f"{tag} Lendo {raster_name}...", flush=True)

        gdal_dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if not gdal_dataset:
            result_queue.put((gpu_id, raster_path, False, "Falha ao abrir raster"))
            return

        num_bandas = gdal_dataset.RasterCount
        largura = gdal_dataset.RasterXSize
        altura = gdal_dataset.RasterYSize
        geotransform = gdal_dataset.GetGeoTransform()
        projection = gdal_dataset.GetProjection()
        sr_scale = 4
        n_images = num_bandas // 3

        # Validar imagens
        for idx in imagens_selecionadas:
            if idx < 0 or idx >= n_images:
                result_queue.put((gpu_id, raster_path, False,
                                  f"Índice {idx} inválido (raster tem {n_images} imagens)"))
                return

        imagens_stack = len(imagens_selecionadas)
        print(f"{tag} {raster_name}: {largura}x{altura}, {num_bandas} bandas, "
              f"usando imagens {imagens_selecionadas}", flush=True)

        raster_data = gdal_dataset.ReadAsArray()
        gdal_dataset = None

        band_indices = []
        for img_idx in imagens_selecionadas:
            band_indices.extend([img_idx * 3, img_idx * 3 + 1, img_idx * 3 + 2])
        selected_bands = raster_data[band_indices]
        del raster_data

        # Grid de tiles
        passo = max(1, int(tamanho_tile * (1 - sobreposicao)))
        num_tiles_x = (largura - tamanho_tile) // passo + 1
        num_tiles_y = (altura - tamanho_tile) // passo + 1
        while (num_tiles_x - 1) * passo + tamanho_tile < largura:
            num_tiles_x += 1
        while (num_tiles_y - 1) * passo + tamanho_tile < altura:
            num_tiles_y += 1

        # Coletar tiles válidos
        tiles_info = []
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                x_inicio = tile_x * passo
                y_inicio = tile_y * passo
                if x_inicio >= largura or y_inicio >= altura:
                    continue
                largura_efetiva = min(tamanho_tile, largura - x_inicio)
                altura_efetiva = min(tamanho_tile, altura - y_inicio)
                if largura_efetiva < tamanho_tile / 2 or altura_efetiva < tamanho_tile / 2:
                    continue
                tile_data = selected_bands[:, y_inicio:y_inicio+altura_efetiva,
                                           x_inicio:x_inicio+largura_efetiva]
                if np.all(tile_data == 0):
                    continue
                tiles_info.append((x_inicio, y_inicio, largura_efetiva, altura_efetiva))

        total_tiles = len(tiles_info)
        print(f"{tag} {total_tiles} tiles válidos, batch_size={batch_size}", flush=True)

        # Mosaico
        largura_sr = largura * sr_scale
        altura_sr = altura * sr_scale
        mosaic_data = np.zeros((3, altura_sr, largura_sr), dtype=np.float32)
        weight_sum = np.zeros((altura_sr, largura_sr), dtype=np.float32)
        mascara_cache = {}

        # Inferência
        t_infer = time.time()
        processed = 0

        with torch.no_grad():
            for batch_start in range(0, total_tiles, batch_size):
                batch_end = min(batch_start + batch_size, total_tiles)
                batch_tiles = tiles_info[batch_start:batch_end]
                cur_batch_size = len(batch_tiles)

                tensors = []
                for (x_ini, y_ini, w_ef, h_ef) in batch_tiles:
                    tile_data = selected_bands[:, y_ini:y_ini+h_ef, x_ini:x_ini+w_ef]
                    if w_ef < tamanho_tile or h_ef < tamanho_tile:
                        padded = np.zeros((tile_data.shape[0], tamanho_tile, tamanho_tile),
                                          dtype=tile_data.dtype)
                        padded[:, :h_ef, :w_ef] = tile_data
                        tile_data = padded
                    tensor = prepare_tensor_from_tile(tile_data, n_lr_images, device)
                    tensors.append(tensor)

                batch_tensor = torch.cat(tensors, dim=0)
                outputs = model(batch_tensor)
                outputs = torch.clamp(outputs, 0, 1)
                outputs_np = outputs.cpu().numpy()

                for i in range(cur_batch_size):
                    x_ini, y_ini, w_ef, h_ef = batch_tiles[i]
                    sr_out = outputs_np[i]
                    sr_w_ef = w_ef * sr_scale
                    sr_h_ef = h_ef * sr_scale
                    sr_out = sr_out[:, :sr_h_ef, :sr_w_ef]

                    pos_x = x_ini * sr_scale
                    pos_y = y_ini * sr_scale
                    h_final = min(sr_h_ef, altura_sr - pos_y)
                    w_final = min(sr_w_ef, largura_sr - pos_x)
                    if h_final <= 0 or w_final <= 0:
                        continue
                    sr_out = sr_out[:, :h_final, :w_final]

                    mask_key = (w_final, h_final)
                    if mask_key not in mascara_cache:
                        mascara_cache[mask_key] = criar_mascara_ponderacao(
                            h_final, w_final, tipo_ponderacao)
                    mascara = mascara_cache[mask_key]

                    mosaic_data[:, pos_y:pos_y+h_final, pos_x:pos_x+w_final] += \
                        sr_out * mascara[np.newaxis, :, :]
                    weight_sum[pos_y:pos_y+h_final, pos_x:pos_x+w_final] += mascara

                processed += cur_batch_size
                elapsed = time.time() - t_infer
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total_tiles - processed) / speed if speed > 0 else 0
                print(f"{tag} [{processed}/{total_tiles}] {speed:.0f} tiles/s | ETA: {eta:.0f}s",
                      flush=True)

        del selected_bands

        # Normalizar
        weight_sum = np.maximum(weight_sum, 1e-10)
        mosaic_data /= weight_sum[np.newaxis, :, :]
        mosaic_data = np.clip(mosaic_data * 255, 0, 255).astype(np.uint8)
        del weight_sum

        # Salvar GeoTIFF
        print(f"{tag} Escrevendo {output_path}...", flush=True)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_path, largura_sr, altura_sr, 3, gdal.GDT_Byte,
                               options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])

        sr_geotransform = list(geotransform)
        sr_geotransform[1] = geotransform[1] / sr_scale
        sr_geotransform[5] = geotransform[5] / sr_scale
        dst_ds.SetGeoTransform(sr_geotransform)
        dst_ds.SetProjection(projection)

        for b in range(3):
            dst_ds.GetRasterBand(b + 1).WriteArray(mosaic_data[b])
        dst_ds = None

        elapsed_total = time.time() - t_total
        print(f"{tag} CONCLUÍDO: {raster_name} → {os.path.basename(output_path)} "
              f"({total_tiles} tiles, {elapsed_total:.1f}s, {total_tiles/elapsed_total:.0f} tiles/s)",
              flush=True)

        result_queue.put((gpu_id, raster_path, True, f"{elapsed_total:.1f}s"))

    except Exception as e:
        result_queue.put((gpu_id, raster_path, False, str(e)))
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline SR Multi-GPU para servidor Linux")

    # Entrada: arquivos individuais ou diretório
    parser.add_argument('-input', type=str, nargs='*', default=None,
                        help="Caminhos dos GeoTIFFs de entrada")
    parser.add_argument('-input_dir', type=str, default=None,
                        help="Diretório com GeoTIFFs (alternativa a -input)")
    parser.add_argument('-output', type=str, nargs='*', default=None,
                        help="Caminhos de saída (1 por input)")
    parser.add_argument('-output_dir', type=str, default=None,
                        help="Diretório de saída (alternativa a -output)")

    # Modelo
    parser.add_argument('-opt', type=str, default='config_server.yml',
                        help="Arquivo YAML de configuração do modelo")

    # Parâmetros do pipeline
    parser.add_argument('-imagens', type=int, nargs='+', default=[0, 1, 2, 3],
                        help="Índices das imagens Sentinel (default: 0 1 2 3)")
    parser.add_argument('-tile_size', type=int, default=32,
                        help="Tamanho dos tiles (default: 32)")
    parser.add_argument('-overlap', type=float, default=0.5,
                        help="Sobreposição entre tiles (default: 0.5)")
    parser.add_argument('-batch_size', type=int, default=256,
                        help="Tiles por batch por GPU (default: 256)")
    parser.add_argument('-ponderacao', type=str, default='linear',
                        choices=['linear', 'gaussiana', 'coseno'],
                        help="Tipo de ponderação (default: linear)")

    # GPU
    parser.add_argument('-gpus', type=int, nargs='*', default=None,
                        help="IDs das GPUs a usar (default: todas disponíveis)")

    args = parser.parse_args()

    # Detectar GPUs
    n_gpus_available = torch.cuda.device_count()
    if n_gpus_available == 0:
        print("ERRO: Nenhuma GPU CUDA encontrada.")
        return

    if args.gpus is not None:
        gpu_ids = args.gpus
    else:
        gpu_ids = list(range(n_gpus_available))

    print(f"GPUs disponíveis: {n_gpus_available}")
    print(f"GPUs que serão usadas: {gpu_ids}")
    for gid in gpu_ids:
        name = torch.cuda.get_device_name(gid)
        mem = torch.cuda.get_device_properties(gid).total_memory / 1024**3
        print(f"  GPU {gid}: {name} ({mem:.0f} GB)")

    # Montar lista de inputs/outputs
    if args.input_dir:
        input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.tif')))
        if not input_files:
            print(f"ERRO: Nenhum .tif encontrado em {args.input_dir}")
            return
    elif args.input:
        input_files = args.input
    else:
        print("ERRO: Forneça -input ou -input_dir")
        return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_files = [
            os.path.join(args.output_dir,
                         os.path.splitext(os.path.basename(f))[0] + '_super.tif')
            for f in input_files
        ]
    elif args.output:
        output_files = args.output
        if len(output_files) != len(input_files):
            print(f"ERRO: {len(input_files)} inputs mas {len(output_files)} outputs")
            return
    else:
        # Default: mesmo diretório com sufixo _super
        output_files = [
            os.path.splitext(f)[0] + '_super.tif' for f in input_files
        ]

    print(f"\nRasters a processar: {len(input_files)}")
    for inp, out in zip(input_files, output_files):
        print(f"  {os.path.basename(inp)} → {os.path.basename(out)}")

    # Processar em rounds de N GPUs
    t_total = time.time()
    result_queue = Queue()
    total_ok = 0
    total_fail = 0

    for round_start in range(0, len(input_files), len(gpu_ids)):
        round_end = min(round_start + len(gpu_ids), len(input_files))
        round_files = list(zip(input_files[round_start:round_end],
                               output_files[round_start:round_end]))

        print(f"\n{'='*60}")
        print(f"ROUND {round_start // len(gpu_ids) + 1}: "
              f"processando {len(round_files)} raster(s) em paralelo")
        print(f"{'='*60}")

        processes = []
        for i, (inp, out) in enumerate(round_files):
            gpu_id = gpu_ids[i]
            p = Process(
                target=process_single_raster,
                args=(gpu_id, inp, out, args.opt,
                      args.imagens, args.tile_size, args.overlap,
                      args.batch_size, args.ponderacao, result_queue)
            )
            p.start()
            processes.append(p)

        # Aguardar todos do round terminarem
        for p in processes:
            p.join()

        # Coletar resultados
        while not result_queue.empty():
            gpu_id, raster, success, msg = result_queue.get()
            name = os.path.basename(raster)
            if success:
                total_ok += 1
                print(f"  OK: {name} (GPU {gpu_id}, {msg})")
            else:
                total_fail += 1
                print(f"  FALHOU: {name} (GPU {gpu_id}): {msg}")

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"TUDO CONCLUÍDO em {elapsed:.1f}s")
    print(f"  Sucesso: {total_ok}, Falhas: {total_fail}")
    print(f"{'='*60}")


if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method('spawn', force=True)
    main()

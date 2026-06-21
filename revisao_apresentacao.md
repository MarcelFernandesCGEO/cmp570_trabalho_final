# Revisão da apresentação — perguntas e respostas da banca

Trabalho final CMP570 — *Superresolução no Domínio de Frequências: Fine-tuning do
SATLAS com Focal Frequency Loss para Correção do Gap Espectral Planet–Aerofoto*.
Autor: Marcel Fernandes Gomes.

Apresentação ~10 min + perguntas. Abaixo, perguntas prováveis da banca agrupadas por
tema, com respostas curtas (o que dizer) e o respaldo nos dados. Números conferidos
contra `resultados/evaluation/results_table.md` (rodada oficial, 20.000 iters, 1.476
chips) em 2026-06-21.

---

## 1. Conceito e teoria

**P1. O que é o "gap espectral" Planet → aerofoto, em uma frase?**
É o déficit de energia nas **altas frequências** do espectro de potência (PSD) das
imagens PlanetScope (~3 m/px) frente às aerofotos (~0,35–0,5 m/px): a aerofoto tem
detalhe fino (bordas, texturas) que aparece como energia em alta frequência, e a
imagem de satélite não. O gap é visível como uma queda mais acentuada da PSD radial
do Planet nas frequências médias–altas.

**P2. Por que medir no domínio de frequências e não só com PSNR/SSIM?**
Porque SSIM e PSNR são métricas **espaciais** dominadas pelas baixas frequências
(maior energia → maior contribuição ao MSE). Um modelo pode ter SSIM alto e ainda
assim entregar texturas finas borradas. O **PSD_L2** mede diretamente a distância
espectral ao GT, que é exatamente o que a FFL tenta corrigir. No nosso resultado, o
ganho da FFL aparece **só** no PSD_L2 (−7,2%) — SSIM/cPSNR mal se mexem.

**P3. O que é a Focal Frequency Loss e qual o seu mecanismo?**
É uma *loss* que calcula a DFT 2D da saída e do GT e penaliza o erro componente a
componente no espectro: `L_FFL = (1/MN) Σ w[u,v]·|F̂[u,v] − F[u,v]|²`. O "focal" está
no peso adaptativo `w[u,v] = |F̂−F|^α / E[|F̂−F|^α]`, recalculado a cada iteração com
`.detach()` (não entra no *backprop*). Frequências que o modelo já acerta recebem
peso baixo; as que ele erra (tipicamente altas) recebem peso maior e mais gradiente.
É um reequilíbrio do gradiente que o L1 enviesa para baixas frequências.

**P4. Por que `norm='ortho'` na FFT?**
Porque a DFT ortonormal preserva energia (Parseval: ‖F‖² = ‖f‖²). Assim a escala da
loss espectral é comparável à do espaço de pixel, e com `w[u,v]=1` a FFL coincide com
o MSE no espaço de pixel — o mecanismo focal é justamente o que quebra essa
equivalência ao priorizar frequências específicas.

**P5. Como isso se conecta com o que foi visto em aula?**
Diretamente com as Aulas 12–15: a DFT 2D (`torch.fft.fft2`) é o núcleo da FFL (Aulas
12/13); o gap espectral é um problema de amostragem/aliasing — Planet e Sentinel
amostram a mesma cena com bandas distintas (Aula 14); e o Teorema da Convolução
(Aula 15) liga a PSF do sensor à multiplicação no domínio de frequência. A FFL é uma
minimização de erro numa base ortogonal (Fourier) em vez do espaço de pixel.

---

## 2. Metodologia e protocolo

**P6. Qual é exatamente o modelo de partida?**
O gerador do **SATLAS** (`SSR_RRDBNet`, variante do ESRGAN/RRDBNet com 23 blocos RRDB,
64 features), com entrada multi-frame: 16 imagens × 3 bandas TCI = 48 canais, saída
RGB 4×. Pesos públicos `esrgan_16S2.pth` (Sentinel-2 → NAIP). O *fine-tuning* parte
desses pesos com pares Planet → aerofoto.

**P7. O que é um *ablation study* e por que ele é o desenho certo aqui?**
É variar **um único fator** (aqui, o λ da FFL) mantendo todo o resto fixo, para
isolar o efeito desse fator. Comparo baseline (sem FFL), λ=0,1 e λ=1,0, com o **mesmo
ponto de partida, mesmos dados e mesmo orçamento de treino (20k iters)**. Qualquer
diferença nas métricas é atribuível à FFL, não a confounds.

**P8. Por que fixar o checkpoint no iter final (20k) e não o "melhor" por SSIM?**
Para evitar viés: cada variante poderia ter o pico em iterações diferentes, e escolher
"o melhor de cada" favoreceria artificialmente alguma. Mesmo orçamento, mesmo ponto de
leitura → comparação justa.

**P9. O que são as três métricas e por que reportar as três?**
SSIM (estrutura, luminância/contraste local), cPSNR (PSNR com **ajuste afim por canal**
— tolera offset/escala radiométrica entre sensores diferentes) e PSD_L2 (distância L2
no log₁₀ do PSD radial médio vs GT). As três cobrem eixos distintos: estrutura,
fidelidade de pixel cross-sensor e fidelidade espectral. A história só fecha vendo as
três juntas: a FFL ganha em PSD_L2 sem destruir SSIM/cPSNR.

**P10. Por que cPSNR em vez de PSNR puro?**
Porque Planet e aerofoto são sensores diferentes, com offset e ganho radiométrico
distintos. O PSNR puro penalizaria essa diferença de tom como se fosse erro de
reconstrução. O cPSNR encontra a transformação afim `a·ŷ+b` por canal que minimiza o
MSE antes de medir — isola o que é de fato qualidade de reconstrução.

---

## 3. Resultados

**P11. Qual é o resultado principal, em uma frase?**
A FFL com λ=1,0 **reduz o PSD_L2 em 7,2%** vs baseline (0,2574 → 0,2389), com custo
desprezível de SSIM (−0,006) e cPSNR (−0,06 dB): a *loss* espectral de fato empurra o
gerador a recuperar frequências altas que o L1/perceptual suavizam.

**P12. Por que λ=0,1 não funcionou?**
Porque o gap Planet→aerofoto é grande; com peso baixo a contribuição da FFL ao
gradiente total é pequena demais para o mecanismo focal "vencer" o L1. λ=0,1 ficou
inclusive com PSD_L2 levemente pior que o baseline (0,2638) — ruído, na prática sem
efeito. Só com λ=1,0 (mesmo peso do L1) a FFL atua.

**P13. O que explica o trade-off SSIM ↓ / PSD_L2 ↓ em λ=1,0?**
São objetivos parcialmente conflitantes. O L1 favorece alinhamento estrutural
pixel-a-pixel (estrutura grossa, SSIM); a FFL favorece *casar o espectro*, o que
introduz textura de alta frequência que nem sempre cai no lugar exato do pixel do GT
(custa SSIM). O modelo "troca" um pouco de SSIM por bastante fidelidade espectral —
trade-off favorável quando o objetivo é detalhe fino.

**P14. Os ganhos são grandes? 7,2% é significativo?**
É um ganho consistente na métrica que importa para o problema (fidelidade espectral),
obtido **sem** custo apreciável nas demais. Não afirmo significância estatística
formal (não há intervalo de confiança entre runs); o que mostro é o efeito direcional
claro e coerente com o mecanismo teórico da FFL e com a literatura (Jiang et al.).

**P15. A avaliação é no mesmo conjunto de treino?**
Sim — não há split separado de teste; a avaliação é nos 1.476 chips de treino/val.
Isso é uma limitação para *generalização*, mas **válido para o ablation**: o objetivo é
medir o **efeito espectral relativo** entre variantes treinadas identicamente, não a
performance absoluta em dados novos. Todas as variantes veem exatamente os mesmos chips.

---

## 4. Reprodutibilidade e honestidade científica

**P16. A reprodução não confirmou o resultado. Como você explica isso?**
Sou transparente sobre isso: a rodada **oficial** (servidor Linux, 20.000 iters) deu o
ganho de 7,2%. Uma reprodução posterior no Windows com **apenas 5.000 iters**
(`resultados/zero_shot_5k/`) **não** reproduziu o ganho — a FFL λ=1,0 ficou levemente
acima do baseline em PSD_L2. A leitura é de **orçamento de treino**: o peso focal da
FFL se acumula ao longo das iterações; com 1/4 do treino o efeito não amadurece, e o
modelo inteiro está pior (SSIM ~0,59 vs ~0,63). O resultado reportado é o de 20k iters;
para replicar é preciso treinar 20k.

**P17. Por que então não rodou 20k de novo no Windows para confirmar?**
Custo de GPU e tempo: a máquina Linux original (com o checkpoint CP06) foi perdida, e o
pipeline foi reescrito para partir do pretrained público. A rodada de 5k foi uma
validação rápida do pipeline reescrito, não uma tentativa de replicar o número. É a
direção imediata de trabalho futuro: re-treinar 20k no ambiente atual.

**P18. O que está e o que não está no repositório?**
Estão: código (FFL standalone, avaliação), configs do ablation, scripts do pipeline,
**ambas** as tabelas de resultados (20k oficial e 5k), relatório, slides e os 16 TIFs
Planet de amostra. Não estão (por tamanho / restrição de redistribuição): a aerofoto GT
(~844 MB), as saídas SR (~400 MB) e os pesos — documentados em `dados/PROVENIENCIA.md`.

---

## 5. Limitações e trabalhos futuros

**P19. Quais as principais limitações?**
(i) Avaliação no conjunto de treino (sem split de teste); (ii) ausência de repetições
para significância estatística; (iii) o resultado-chave (20k) não foi re-confirmado no
ambiente atual (5k não reproduziu); (iv) uma única região geográfica.

**P20. Trabalhos futuros?**
Re-treinar 20k no ambiente atual para confirmar o ganho; varrer λ intermediários (0,3,
0,5) para mapear o trade-off; reservar um split de teste independente; ampliar a área
(a "rodada 2" com ~9.360 chips / ~224 km² foi preparada mas não avaliada); comparar com
outros métodos de SR guiados por frequência.

**P21. Por que não usar Sentinel-2 direto, já que o SATLAS foi treinado nele?**
Porque o objetivo aplicado é justamente usar **Planet** (maior disponibilidade
comercial/temporal) onde o Sentinel não basta. O domain shift Planet→S2 é parte do
problema — e o *fine-tuning* + FFL é a forma de atacá-lo no domínio espectral.

---

## 6. Perguntas de fundamentação ("difíceis")

**P22. Os pesos `w[u,v]` participam do gradiente?**
Não. São calculados com `.detach()` a partir do erro da iteração anterior — entram como
**constante** na loss da iteração atual. Isso evita um termo de segunda ordem instável e
é fiel à formulação de Jiang et al.

**P23. A FFL não é só um MSE espectral disfarçado?**
Com `w=1` e `norm='ortho'`, sim — pela igualdade de Parseval seria equivalente ao MSE no
pixel. O que a distingue é o **peso adaptativo focal**: ele quebra a equivalência ao
realocar o gradiente para as frequências mal recuperadas, que num MSE plano seriam
ofuscadas pela energia das baixas frequências.

**P24. Qual o papel do α?**
α controla a agressividade do foco: α=1 → peso linear no erro; α>1 → penaliza
desproporcionalmente as frequências mais difíceis. Usei α=1,0 nas duas variantes com
FFL (varia-se só o λ no ablation).

**P25. Por que o gerador é multi-frame (16 imagens)?**
Porque o SATLAS explora informação **sub-pixel** entre múltiplas datas: várias passagens
ligeiramente deslocadas trazem detalhe que uma única imagem não tem. Empilha 16 frames
(48 canais) e produz uma saída SR. No nosso caso isso é entrada fixa do modelo herdado.

**P26. Qual é, afinal, a contribuição original?**
Três coisas: (i) integração **retrocompatível** da FFL ao SATLAS/BasicSR (2 patches,
configs sem `ffl_opt` ficam idênticas ao original); (ii) um *ablation* controlado do
efeito espectral da FFL no domínio Planet→aerofoto; (iii) o uso do **PSD_L2** como
métrica que expõe um ganho (−7,2%) invisível a SSIM/PSNR.

---

## 7. Checklist rápido antes de apresentar

- [ ] Saber de cor os 3 números-âncora: **0,2574 → 0,2389** (PSD_L2, −7,2%),
      **ΔSSIM = −0,006**, **λ=0,1 não funciona**.
- [ ] Ter clara a frase-mecanismo: *a FFL realoca o gradiente, dominado pelo L1 nas
      baixas frequências, para as altas mal recuperadas*.
- [ ] Saber explicar o trade-off SSIM↓ / PSD_L2↓ (espectral vs espacial).
- [ ] Estar pronto para a pergunta da reprodução 5k vs 20k — responder com honestidade
      (orçamento de treino) e apontar como trabalho futuro.
- [ ] Tabelas dos slides = `resultados/evaluation/results_table.md` (conferido).
- [ ] Nome do autor padronizado: **Marcel Fernandes Gomes**.

---

## 8. Glossário

### Domínio de frequências
- **DFT 2D.** Transformada Discreta de Fourier de uma imagem; `torch.fft.fft2`.
  Decompõe a imagem em frequências espaciais.
- **PSD (Power Spectral Density).** `|F[u,v]|²`; energia por frequência. A **PSD radial**
  média integra em anéis `r=√(u²+v²)`, resumindo o espectro independentemente de orientação.
- **Gap espectral.** Déficit de energia em altas frequências do Planet vs aerofoto.
- **`norm='ortho'`.** Normalização ortonormal da FFT; preserva energia (Parseval).
- **Aliasing.** Distorção por subamostragem; fundo do gap entre sensores (Aula 14).

### Loss e treino
- **FFL (Focal Frequency Loss).** Erro ponderado no espectro; peso focal adaptativo
  `w=|F̂−F|^α/E[·]`, com `.detach()`. Jiang et al., ICCV 2021.
- **L1 / Perceptual (VGG) / GAN.** Componentes da loss do ESRGAN: erro absoluto de pixel,
  similaridade em features de uma rede, e termo adversarial.
- **λ_FFL.** Peso da FFL na loss total (0 / 0,1 / 1,0 no ablation).
- **α.** Expoente do mecanismo focal (1,0 aqui).
- **`.detach()`.** Remove o tensor do grafo de autograd → os pesos não geram gradiente.
- **Ablation study.** Variar um fator de cada vez para isolar seu efeito.
- **iter / total_iter.** Iteração de treino; 20.000 na rodada oficial.

### Modelo e dados
- **SATLAS / SATLAS-SR.** Modelo de SR de sensoriamento remoto da AllenAI (paper
  *Zooming Out on Zooming In*), Sentinel-2 → NAIP, 4×.
- **ESRGAN / RRDBNet / RRDB.** Backbone do gerador; RRDB = *Residual-in-Residual Dense
  Block*; a rede usa 23 RRDBs. `SSR_RRDBNet` = variante multi-frame do SATLAS.
- **Multi-frame.** Entrada de N=16 imagens (48 canais) explorando informação sub-pixel.
- **Fine-tuning.** Reajuste dos pesos pré-treinados em dados/objetivo novos.
- **TCI.** *True Color Image*; produto RGB do Planet/Sentinel.
- **PlanetScope.** Constelação comercial ~3 m/px; entrada LR.
- **Aerofoto GT.** Imagem aérea ~0,35–0,5 m/px (Convênio Estado RS / Exército); HR.
- **Chip.** Recorte fixo: 32×32 px (LR Planet) ↔ 128×128 px (HR aerofoto), escala 4×.

### Métricas
- **SSIM.** *Structural Similarity Index*; 0–1, maior melhor.
- **cPSNR.** PSNR com ajuste afim por canal; tolera offset/escala cross-sensor; maior melhor.
- **PSD_L2.** Distância L2 no log₁₀ do PSD radial médio vs GT; **métrica principal**;
  menor melhor.

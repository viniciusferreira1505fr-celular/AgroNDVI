AgroNDVI 1.0.0 — painel, relatórios, histórico e backup

INSTALAÇÃO WINDOWS
1. Extraia o ZIP completo para uma pasta nova.
2. Para preservar dados de versões anteriores, copie para esta pasta os arquivos que você já usa:
   - talhoes_agroNDVI.json
   - historico_ndvi_agroNDVI.csv (se existir)
   Confirme substituir os arquivos de exemplo.
3. Dê duplo clique em iniciar_AgroNDVI.bat.
4. Se o Windows/terminal informar dependências ausentes, execute no terminal da pasta:
   py -m pip install -r requirements.txt
   e depois execute o BAT novamente.

NOVIDADES 0.8.0
- Nova tela Painel e relatórios com indicadores de talhões e histórico.
- Catálogo de talhões em tabela, com exportação CSV.
- Gráfico de histórico NDVI por talhão e métricas de NDVI recente e variação.
- Exportação do histórico filtrado em CSV.
- Relatório PDF com gráfico temporal e tabela dos registros mais recentes.
- Downloads de backup do catálogo, backup automático anterior e histórico completo.
- Backup do catálogo antes de salvar: cópia .bak e cópia datada.
- Backup do histórico CSV antes de atualizar: arquivo .bak.
- Salvamento atômico dos dados mantido.
- Corrigida a identificação do talhão nas séries Sentinel salvas no histórico.
- Mantidas as funções de desenho/gestão de talhões, GPS manual por pontos, Sentinel-2, NDVI e GeoTIFF.

ARQUIVOS LOCAIS
- talhoes_agroNDVI.json: cadastro de polígonos/talhões.
- talhoes_agroNDVI.json.bak: cópia anterior ao salvamento mais recente.
- talhoes_agroNDVI.json.backup_YYYYMMDD_HHMMSS: cópias datadas antes de alterações.
- historico_ndvi_agroNDVI.csv: histórico de processamento de séries.
- historico_ndvi_agroNDVI.csv.bak: histórico anterior à atualização.

NOTAS
- Mosaico Esri do mapa é imagem de referência, não satélite em tempo real.
- Sentinel-2 tem revisita nominal de cerca de 5 dias; nuvens e processamento podem limitar cenas úteis.
- NDVI não diagnostica sozinho doenças, nematoides ou causas de estresse.
- Faça cópias periódicas dos arquivos de dados em outro computador/nuvem.
- Esta versão é uma atualização de MVP; valide os resultados com dados e procedimentos da sua pesquisa antes de usar em relatórios técnicos oficiais.


NOVO NA VERSAO 0.8.0
- Area Pesquisa experimental para inserir/importar dados de parcelas.
- Resumo por tratamento, grafico de medias e boxplot.
- ANOVA de uma via exploratoria (p-valor via scipy).
- Exportacao de dados e resumo em CSV.
- Baixar CSV modelo.
Nota: nesta versao, os valores NDVI por parcela devem ser informados/importados; a extracao automatica por poligonos de parcelas sera implementada posteriormente.


NOVOS RECURSOS CONSOLIDADOS 1.0.0
- Ensaios experimentais persistentes (JSON local)
- Delineamentos DIC e DBC, ANOVA exploratória com efeito de bloco quando informado
- Tukey HSD quando aplicável, CV experimental estimado, tabelas e exportações CSV
- Importação de GeoJSON para o catálogo de talhões
- Backup completo ZIP e restauração com cópia de segurança pré-restauração
- Correção de navegação do módulo de talhões

NOTAS IMPORTANTES
- O mosaico Esri é imagem de referência, não satélite ao vivo.
- Sentinel-2 é consultado por catálogo e depende de disponibilidade, nuvens e conexão.
- GPS do navegador registra pontos sob solicitação; rastreamento contínuo em segundo plano não é garantido.
- Estatística é apoio: confirme delineamento, pressupostos, unidade experimental e resíduos antes de interpretar.
- A extração automática de NDVI por parcelas individuais e o buffer de bordadura ainda são itens para uma fase futura; esta versão usa NDVI de área/talhão e dados experimentais importados/registrados.


RECURSO GEoespacial experimental
- Exportar NDVI georreferenciado como GeoTIFF.
- Extrair estatísticas de NDVI por parcela usando GeoJSON de parcelas + raster NDVI.
- Resumo por tratamento e exportação CSV.
- Para parcelas pequenas, pixels Sentinel-2 de 10 m podem ser mistos; use bordadura/área útil e interprete com cautela.

VERSÃO 1.0.1 — MELHORIAS VISUAIS
- Mapa NDVI contínuo com download em PNG.
- Mapa de zonas de vigor NDVI e download em PNG.
- Distribuição por classes de vigor: pixels, área estimada e percentual.
- Gráfico de distribuição de zonas com exportação PNG.
- Gráficos experimentais de médias ± erro padrão e boxplot.
- Download dos gráficos experimentais em PNG.
- As classes de NDVI são faixas visuais genéricas e não limites universais agronômicos.


VERSÃO 1.0.2 — ANÁLISE POR FOTO RGB
- Captura de foto pela câmera do celular via Streamlit.
- Upload de uma ou várias fotos.
- Estimativa de cobertura verde, pixels amarelos/marrons, VARI e ExG.
- Máscara visual classificada e gráficos exportáveis em PNG.
- Comparação e exportação CSV entre fotos.
- Atenção: índices RGB estimativos; não são NDVI verdadeiro e dependem de iluminação e enquadramento.


AGRO NDVI 2.0.0 — CONSOLIDAÇÃO DE MÓDULOS
- Diário agronômico local/offline: cultura, estádio, ocorrência, avaliação, observações, coordenadas e fotos.
- Histórico do diário em tabela, com filtros e exportação CSV/JSON.
- Arquivos de fotos do diário armazenados localmente na pasta fotos_agroNDVI.
- Análise exploratória de raiz por segmentação de imagem: área, esqueleto/ comprimento aparente, diagonal do contorno e extremidades (proxy).
- Calibração opcional em mm/pixel para estimativas métricas.
- Gerador de randomização de ensaios DIC e DBC, com exportação CSV.
- Registro de diário e arquivos locais não exigem internet para operações básicas.
- Análise de raízes depende de imagem com fundo contrastante; métricas e ramificações são estimativas e exigem validação.
- O modo offline é local; consulta Sentinel e mapas base continuam exigindo conexão.

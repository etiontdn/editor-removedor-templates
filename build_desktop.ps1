# Script para empacotar o projeto Django com PyInstaller
Write-Host "Iniciando o empacotamento com PyInstaller..."

# Ativa o ambiente virtual se não estiver ativo
if (-Not (Test-Path "Env:VIRTUAL_ENV")) {
    Write-Host "Ativando ambiente virtual..."
    .\venv\Scripts\Activate.ps1
}

# Comando PyInstaller
# --noconfirm: Sobrescreve o diretório dist/ sem perguntar
# --onedir: Cria um diretório com o executável e dependências (melhor para debug inicial)
# --add-data: Inclui as pastas do projeto Django

pyinstaller --noconfirm --onedir `
    --name "Editor e Removedor de Templates" `
    --add-data "core;core" `
    --add-data "limpador/templates;limpador/templates" `
    --add-data "limpador/static;limpador/static" `
    --add-data "db.sqlite3;." `
    --hidden-import "limpador.apps" `
    main.py

Write-Host "Processo concluído! Verifique a pasta 'dist/Editor e Removedor de Templates'."

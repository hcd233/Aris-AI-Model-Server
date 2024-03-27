registry := "mirrors.tencent.com"

base_image := "fibona-mas-base"
image := "fibona-mas"

base_repository := "zyp-fibona-ai/" + base_image
repository := "zyp-fibona-ai/" + image

branch := `git rev-parse --abbrev-ref HEAD`
tag := replace(branch, "/", "-") + "-" + `date +'%y%m%d%H%M'`

platforms := "linux/amd64"

# list all available recipes
default:
    @just --list --justfile {{ justfile() }}

[private]
base_qci QCI_ENV_FILE:
    echo "ARTIFACT_VERSION={{ tag }}" >>"{{ QCI_ENV_FILE }}"
    @echo "Building base docker image"
    docker buildx build --platform={{ platforms }} -f docker/mas_base.dockerfile -t {{ registry }}/{{ repository }}:{{ tag }} .
    docker push {{ registry }}/{{ repository }}:{{ tag }}

qci QCI_ENV_FILE:
    echo "ARTIFACT_VERSION={{ tag }}" >>"{{ QCI_ENV_FILE }}"
    @echo "Building docker image"
    docker buildx build --platform={{ platforms }} -f docker/dockerfile -t {{ registry }}/{{ repository }}:{{ tag }} .
    docker push {{ registry }}/{{ repository }}:{{ tag }}

env:
    poetry install --no-root

grpc:
    poetry shell
    python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. src/grpc/langchain/v1/langchain.proto
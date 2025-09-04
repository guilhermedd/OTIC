# OTIC
Repositório da disciplina **OTIC**.

## 3-SAT Problem Solver

Este projeto implementa um solucionador do problema **3-SAT** usando um algoritmo genético genérico.

### Estrutura do projeto

- **3-SAT/3_sat_conda_env.yml**  
  Arquivo para configurar o ambiente Conda necessário para rodar o solucionador do 3-SAT.

- **data/**  
  Contém os arquivos de entrada para o problema 3-SAT.

- **src/**  
  Implementação do solucionador do 3-SAT e do algoritmo genético genérico.

---

### Executando o solucionador

Exemplo de execução usando Python e argumentos via linha de comando:

```bash
python main.py \
    --num_threads 4 \
    --data_file "data/100.txt" \
    --population_size 30 \
    --dimension 1 \
    --lower_bound 0 \
    --upper_bound 1 \
    --population_type "B"
```

`--num_threads` : Número de threads/processos a serem usados.

`--data_file` : Caminho para o arquivo de entrada .cnf.

`--population_size` : Tamanho da população do algoritmo genético.

`--dimension` : Número de variáveis de cada indivíduo.

`--lower_bound` : Limite inferior para inicialização dos indivíduos.

`--upper_bound` : Limite superior para inicialização dos indivíduos.

`--population_type` : Tipo da população:

   - 'f' → float32

   - 'i' → inteiro

   - 'B' → binário (0 ou 1)
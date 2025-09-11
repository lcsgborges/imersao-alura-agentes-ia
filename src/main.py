if __name__ == '__main__':
    from config import question_model
    
    tests = [
        "Posso reembolsar a internet?",
        "Quero mais 5 dias de trabalho remoto. Como faço?",
        "Posso reembolsar cursos ou treinamentos da Alura?",
        "Quantas capivaras tem no Rio Pinheiros?"
    ]
    
    for question in tests:
        response = question_model(question)
        
        print(f'\nPergunta: {question}\n')
        print(f'\nResposta: {response['answer']}\n')
        if response['context']:
            for c in response['cites']:
                print(f'   - {c}')
        print("\n================================\n")
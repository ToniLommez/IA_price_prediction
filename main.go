package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
)

func readFile(filename string) ([]int, []int, []int, []int, []int) {
	// open file
	f, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	// remember to close the file at the end of the program
	defer f.Close()

	// read the file word by word using scanner
	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)

	preco := make([]int, 0)
	tamanho := make([]int, 0)
	quarto := make([]int, 0)
	banheiro := make([]int, 0)
	suite := make([]int, 0)
	vagas := make([]int, 0)
	tempArray := make([]int, 0)

	for scanner.Scan() {
		temp, _ := strconv.Atoi(scanner.Text())
		tempArray = append(tempArray, temp)
	}

	for i := 0; i < len(tempArray); {
		preco = append(preco, tempArray[i])
		i++
		tamanho = append(tamanho, tempArray[i])
		i++
		quarto = append(quarto, tempArray[i])
		i++
		banheiro = append(banheiro, tempArray[i])
		i++
		suite = append(suite, tempArray[i])
		i++
		vagas = append(vagas, tempArray[i])
		i++
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	return preco, tamanho, quarto, banheiro, vagas
}

func compute_cost(x1_train []float64, x2_train []float64, x3_train []float64, x5_train []float64, w1 float64, w2 float64, w3 float64, w5 float64, b float64, y_train []float64) (float64, float64, float64, float64, float64) {
	m := len(x1_train)
	dj_dw1 := 0.0
	dj_dw2 := 0.0
	dj_dw3 := 0.0
	dj_dw5 := 0.0
	dj_db := 0.0

	for i := 0; i < m; i++ {
		f_wb := w1*x1_train[i] + w2*x2_train[i] + w3*x3_train[i] + w5*x5_train[i] + b
		dj_dw1_t := (f_wb - y_train[i]) * x1_train[i]
		dj_dw2_t := (f_wb - y_train[i]) * x2_train[i]
		dj_dw3_t := (f_wb - y_train[i]) * x3_train[i]
		dj_dw5_t := (f_wb - y_train[i]) * x5_train[i]
		dj_db_t := (f_wb - y_train[i])
		dj_dw1 += dj_dw1_t
		dj_dw2 += dj_dw2_t
		dj_dw3 += dj_dw3_t
		dj_dw5 += dj_dw5_t
		dj_db += dj_db_t
	}
	dj_dw1 = dj_dw1 / float64(m)
	dj_dw2 = dj_dw2 / float64(m)
	dj_dw3 = dj_dw3 / float64(m)
	dj_dw5 = dj_dw5 / float64(m)
	dj_db = dj_db / float64(m)

	return dj_dw1, dj_dw2, dj_dw3, dj_dw5, dj_db
}

func gradient_descent(x1_train []float64, x2_train []float64, x3_train []float64, x5_train []float64, y_train []float64) (float64, float64, float64, float64, float64) {
	w1 := 0.0
	w2 := 0.0
	w3 := 0.0
	w5 := 0.0
	w1_t := 0.0
	w2_t := 0.0
	w3_t := 0.0
	w5_t := 0.0
	b := 0.0
	b_t := 0.0
	alpha := 1.0e-4
	limit := 10000000

	for i := 0; i < limit; i++ {
		dj_dw1, dj_dw2, dj_dw3, dj_dw5, dj_db := compute_cost(x1_train, x2_train, x3_train, x5_train, w1, w2, w3, w5, b, y_train)

		w1_t = w1 - alpha*dj_dw1
		w2_t = w2 - alpha*dj_dw2
		w3_t = w3 - alpha*dj_dw3
		w5_t = w5 - alpha*dj_dw5
		b_t = b - alpha*dj_db
		w1 = w1_t
		w2 = w2_t
		w3 = w3_t
		w5 = w5_t
		b = b_t
	}

	return w1, w2, w3, w5, b
}

func normalizar(valor []int) ([]float64, float64) {
	maior := valor[0]
	for i := 1; i < len(valor); i++ {
		if maior < valor[i] {
			maior = valor[i]
		}
	}
	normalizado := make([]float64, 0)
	if valor[0] < 1000 {
		for i := 0; i < len(valor); i++ {
			normalizado = append(normalizado, float64(valor[i]))
		}
		return normalizado, 1.0
	}
	for i := 0; i < len(valor); i++ {
		normalizado = append(normalizado, float64(valor[i])/(float64(maior)))
	}
	return normalizado, float64(maior)
}

func main() {
	fmt.Print("\nCalculando Valores . . .\nCaso demore a iniciar aperte ENTER\n")
	preco, tamanho, quarto, banheiro, vagas := readFile("Treinamento.txt")

	preco_norm, preco_const := normalizar(preco)
	tamanho_norm, tamanho_const := normalizar(tamanho)
	quarto_norm, quarto_const := normalizar(quarto)
	banheiro_norm, banheiro_const := normalizar(banheiro)
	vaga_norm, vaga_const := normalizar(vagas)

	w1_f, w2_f, w3_f, w5_f, b_f := gradient_descent(tamanho_norm, quarto_norm, banheiro_norm, vaga_norm, preco_norm)
	fmt.Printf("Valores Encontrados na regressao: %.15f , %.15f , %.15f , %.15f , %.15f\n", w1_f, w2_f, w3_f, w5_f, b_f)
	fmt.Printf("Formula = %.2f * x1 + %.2f * x2 + %.2f * x3 + %.2f * x5 + %.2f * %.2f\n\n", w1_f, w2_f, w3_f, w5_f, b_f, preco_const)

	var tamanhoPredc float64
	var quartoPredc float64
	var banheiroPredc float64
	var vagaPredc float64
	for i := 1; i != 0; {
		fmt.Printf("\nDigite os valores para predizer o tamanho do apartamento:\n")
		fmt.Printf("Tamanho m²-> ")
		fmt.Scanln(&tamanhoPredc)
		fmt.Printf("Quartos -> ")
		fmt.Scanln(&quartoPredc)
		fmt.Printf("Banheiros -> ")
		fmt.Scanln(&banheiroPredc)
		fmt.Printf("Vagas -> ")
		fmt.Scanln(&vagaPredc)

		tamanhoPredc = tamanhoPredc / tamanho_const
		quartoPredc = quartoPredc / quarto_const
		banheiroPredc = banheiroPredc / banheiro_const
		vagaPredc = vagaPredc / vaga_const
		prediction := ((w1_f * tamanhoPredc) + (w2_f * quartoPredc) + (w3_f * banheiroPredc) + (w5_f * vagaPredc) + b_f) * preco_const

		fmt.Printf("price: %d Reais\n\n", int(prediction))

		fmt.Printf("Gostaria de calcular outro?\n0 = nao\n1 = sim\n-> ")
		fmt.Scanln(&i)
	}
}

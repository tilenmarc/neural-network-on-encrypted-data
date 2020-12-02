/*
Copyright (c) 2018 XLAB d.o.o

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"fmt"
	"github.com/pkg/errors"
	"github.com/fentec-project/bn256"
	quad "github.com/fentec-project/gofe/quadratic"
	sample "github.com/fentec-project/gofe/sample"
	data "github.com/fentec-project/gofe/data"
	"github.com/fentec-project/neural-network-on-encrypted-data/utils"
	"math/big"
	"time"
)

// This is a demonstration of how the SGP FE scheme for evaluation of
// quadratic multivariate polynomials can be used
// to evaluate a machine learning function on encrypted data.
//
// First, we assume that we learned a function for recognizing numbers
// from images using TensorFlow, and saved the parameters to files
// mat_valid.txt, mat_diag.txt and mat_proj.txt, which we put in the
// testdata folder.
func main() {
	q := quad.NewSGP(1, big.NewInt(2))
	bits := 38
	repeats := 100
	g1gen := new(bn256.G1).ScalarBaseMult(big.NewInt(1))
	g2gen := new(bn256.G2).ScalarBaseMult(big.NewInt(1))
	g := bn256.Pair(g1gen, g2gen)

	bound2 := new(big.Int).Exp(big.NewInt(2), big.NewInt(int64(bits)), nil)
	fmt.Println("precomputing dlog table")
	q.GCalc = q.GCalc.WithBound(bound2)
	start1 := time.Now()
	q.GCalc.Precompute(g)
	elapsed1 := time.Since(start1)
	fmt.Println("precompute took (ms)", elapsed1.Milliseconds())

	mnist(40, "mnist", bits, repeats, q.GCalc.Precomp)
	mnist(128, "f", bits, repeats, q.GCalc.Precomp)
	avg_vec(bits, repeats, q.GCalc.Precomp)
}



func mnist(vecSize int, fileName string, bits int, num_repeats int, precompute map[string]*big.Int) {
	//vecSize := 40
	//vecSize := 128

	// Diagonal matrix
	// number of rows of this matrix represents the number of classes.
	// The function will predict one of these classes.
	diag, err := utils.ReadMatFromFile("testdata/" + fileName + "_mat_diag.txt")
	//diag, err := utils.ReadMatFromFile("testdata/f_mat_diag.txt")
	if err != nil {
		panic(errors.Wrap(err, "error reading diagonal matrix"))
	}
	nClasses := diag.Rows()
	if diag.Cols() != vecSize {
		panic(fmt.Sprintf("diagonal matrix must have %d columns", vecSize))
	}

	// Valid matrix
	// number of rows of this matrix represents the number of examples.
	valid, err := utils.ReadMatFromFile("testdata/" + fileName + "_x_proj.txt")
	//valid, err := utils.ReadMatFromFile("testdata/f_x_proj.txt")
	if err != nil {
		panic(errors.Wrap(err, "error reading valid matrix"))
	}
	if valid.Cols() != vecSize {
		panic(fmt.Sprintf("valid matrix must have %d columns", vecSize))
	}

	// We know that all the values in the matrices are in the
	// interval [-bound, bound].
	bound := big.NewInt(1000000000000000)

	// q is an instance of the FE scheme for quadratic multi-variate
	// polynomials constructed by Sans, Gay, Pointcheval (SGP)
	q := quad.NewSGP(vecSize, bound)

	// we generate a master secret key that we will need for encryption
	// of our data.
	fmt.Println("Generating master secret key...")
	msk, err := q.GenerateMasterKey()
	if err != nil {
		panic(errors.Wrap(err, "error when generating master keys"))
	}



	//// Then, we manipulate the encryption to be the encryption of the
	//// projected data.
	//// Note that this can also be done without knowing the secret key.
	//fmt.Println("Manipulating encryption...")
	//projC := utils.ProjectEncryption(c, proj)
	//
	//fmt.Println("Manipulating secret key...")
	//projSecKey := utils.ProjectSecKey(msk, proj)

	//// We create a new (projected) scheme instance for decrypting
	//newBound := big.NewInt(1500000000)
	//fmt.Println("Creating new (projected) scheme instance for decrypting...")
	//qProj := quad.NewSGP(nVecs, newBound)

	res := make([]*big.Int, nClasses)

	//fmt.Println("Precomputing...")
	//predictedNum := 0 // the predicted number

	//g1gen := new(bn256.G1).ScalarBaseMult(big.NewInt(1))
	//g2gen := new(bn256.G2).ScalarBaseMult(big.NewInt(1))
	//g := bn256.Pair(g1gen, g2gen)

	bound2 := new(big.Int).Exp(big.NewInt(2), big.NewInt(int64(bits)), nil)
	q.GCalc = q.GCalc.WithBound(bound2)
	//start1 := time.Now()
	q.GCalc.Precomp = precompute
	//elapsed1 := time.Since(start1)
	//fmt.Println("precompute took (ms)", elapsed1.Milliseconds())
	q.GCalc = q.GCalc.WithNeg()

	avg := int64(0)
	fmt.Println("Evaluating...")
	for ii:=0; ii<num_repeats; ii++ {
		//fmt.Println(ii)
		// First, we encrypt the data from mat_valid.txt
		// with our master secret key.
		// x = first row of matrix valid
		// y = also the first row of matrix valid
		//fmt.Println("Encrypting...")
		c, err := q.Encrypt(valid[ii], valid[ii], msk)
		if err != nil {
			panic(errors.Wrap(err, "error when encrypting"))
		}
		//start := time.Now()
		//dec, err := q.Decrypt(c, key, f)
		//elapsed := time.Since(start)
		//fmt.Println(elapsed.Milliseconds())
		//fmt.Println("Predicting...")
		sum := int64(0)
		maxValue := new(big.Int).Set(bound2)
		maxValue = maxValue.Neg(maxValue)
		for i := 0; i < nClasses; i++ {
			// We construct a diagonal matrix D that has the elements in the
			// current row of matrix diag on the diagonal.
			D := utils.DiagMat(diag[i])

			// We derive a feKey for obtaining the prediction from the encryption.
			// We will use this feKey for decrypting the final result,
			// e.g. x^T * D * y.
			feKey, err := q.DeriveKey(msk, D)
			if err != nil {
				panic(errors.Wrap(err, "error when deriving FE key"))
			}

			// We decrypt the encryption with the derived key feKey.
			// The result of decryption holds the value of x^T * D * y,
			// which in our case predicts the number from the handwritten
			// image.
			start := time.Now()
			dec, err := q.Decrypt(c, feKey, D)
			elapsed := time.Since(start)
			//fmt.Println(elapsed.Milliseconds())
			res[i] = dec

			sum += elapsed.Milliseconds()
			if err != nil {
				panic(errors.Wrap(err, "error when decrypting"))
			}
			res[i] = dec

			if dec.Cmp(maxValue) > 0 {
				maxValue.Set(dec)
				//predictedNum = i
			}
		}

		//fmt.Println("Time needed:", sum)
		//
		//fmt.Println("Prediction vector:", res)
		//fmt.Println("The model predicts that the number on the image is", predictedNum)
		avg += sum
	}
	//fmt.Println("The model predicts that the number on the image is", predictedNum)

	avg = avg / int64(num_repeats)
	fmt.Println("decrypting 10 images took (ms):", avg)

}

func avg_vec(bits int, num_repeats int, precompute map[string]*big.Int) {
	bound := big.NewInt(100)
	n := 20
	q := quad.NewSGP(n, bound)

	bound2 := new(big.Int).Exp(big.NewInt(2), big.NewInt(int64(bits)), nil)
	q.GCalc = q.GCalc.WithBound(bound2)
	q.GCalc.Precomp = precompute
	q.GCalc = q.GCalc.WithNeg()

	params := [][2]int{{5, 100}, {20, 100}, {1, 500}, {1, 1000}, {20, 1000}}

	for _, e := range params {
		n = e[0]
		q.N = e[0]
		bound.SetInt64(int64(e[1]))
		sampler := sample.NewUniformRange(new(big.Int).Neg(bound), bound)

		avg := int64(0)
		fmt.Println("Evaluating...")
		for ii := 0; ii < num_repeats; ii++ {

			f, err := data.NewRandomMatrix(n, n, sampler)
			if err != nil {
				panic(errors.Wrap(err, "error when generating random matrix: %v"))
			}

			msk, err := q.GenerateMasterKey()
			if err != nil {
				panic(errors.Wrap(err, "error when generating master keys: %v"))
			}

			x, err := data.NewRandomVector(n, sampler)
			if err != nil {
				panic(errors.Wrap(err, "error when generating random vector: %v"))
			}
			y, err := data.NewRandomVector(n, sampler)
			if err != nil {
				panic(errors.Wrap(err, "error when generating random vector: %v"))
			}
			//x[0].Set(big.NewInt(-10))

			c, err := q.Encrypt(x, y, msk)
			if err != nil {
				panic(errors.Wrap(err, "error when encrypting: %v"))
			}

			key, err := q.DeriveKey(msk, f)
			if err != nil {
				panic(errors.Wrap(err, "error when deriving key: %v"))
			}

			check, err := f.MulXMatY(x, y)
			if err != nil {
				panic(errors.Wrap(err, "error when computing x*F*y: %v"))
			}

			start := time.Now()
			dec, err := q.Decrypt(c, key, f)
			elapsed := time.Since(start)
			avg += elapsed.Milliseconds()
			//fmt.Println(elapsed.Milliseconds())
			if err != nil {
				panic(errors.Wrap(err, "error when decrypting: %v"))
			}

			if check.Cmp(dec) != 0 {
				fmt.Println("Decryption wrong")
			}
		}

		avg = avg / int64(num_repeats)
		fmt.Println("result for average vector (ms):", e, avg)
	}
}

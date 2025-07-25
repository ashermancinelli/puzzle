(declare-fun prize_index () (_ BitVec 7))
(declare-fun board () (_ BitVec 64))
(assert (bvult prize_index #b1000000))
(assert (let ((a!1 (bvxor #b0000000
                  (ite (= #b1 ((_ extract 0 0) board)) #b0000000 #b0000000)
                  (ite (= #b1 ((_ extract 1 1) board)) #b0000001 #b0000000)
                  (ite (= #b1 ((_ extract 2 2) board)) #b0000010 #b0000000)
                  (ite (= #b1 ((_ extract 3 3) board)) #b0000011 #b0000000)
                  (ite (= #b1 ((_ extract 4 4) board)) #b0000100 #b0000000)
                  (ite (= #b1 ((_ extract 5 5) board)) #b0000101 #b0000000)
                  (ite (= #b1 ((_ extract 6 6) board)) #b0000110 #b0000000)
                  (ite (= #b1 ((_ extract 7 7) board)) #b0000111 #b0000000)
                  (ite (= #b1 ((_ extract 8 8) board)) #b0001000 #b0000000)
                  (ite (= #b1 ((_ extract 9 9) board)) #b0001001 #b0000000)
                  (ite (= #b1 ((_ extract 10 10) board)) #b0001010 #b0000000)
                  (ite (= #b1 ((_ extract 11 11) board)) #b0001011 #b0000000)
                  (ite (= #b1 ((_ extract 12 12) board)) #b0001100 #b0000000)
                  (ite (= #b1 ((_ extract 13 13) board)) #b0001101 #b0000000)
                  (ite (= #b1 ((_ extract 14 14) board)) #b0001110 #b0000000)
                  (ite (= #b1 ((_ extract 15 15) board)) #b0001111 #b0000000)
                  (ite (= #b1 ((_ extract 16 16) board)) #b0010000 #b0000000)
                  (ite (= #b1 ((_ extract 17 17) board)) #b0010001 #b0000000)
                  (ite (= #b1 ((_ extract 18 18) board)) #b0010010 #b0000000)
                  (ite (= #b1 ((_ extract 19 19) board)) #b0010011 #b0000000)
                  (ite (= #b1 ((_ extract 20 20) board)) #b0010100 #b0000000)
                  (ite (= #b1 ((_ extract 21 21) board)) #b0010101 #b0000000)
                  (ite (= #b1 ((_ extract 22 22) board)) #b0010110 #b0000000)
                  (ite (= #b1 ((_ extract 23 23) board)) #b0010111 #b0000000)
                  (ite (= #b1 ((_ extract 24 24) board)) #b0011000 #b0000000)
                  (ite (= #b1 ((_ extract 25 25) board)) #b0011001 #b0000000)
                  (ite (= #b1 ((_ extract 26 26) board)) #b0011010 #b0000000)
                  (ite (= #b1 ((_ extract 27 27) board)) #b0011011 #b0000000)
                  (ite (= #b1 ((_ extract 28 28) board)) #b0011100 #b0000000)
                  (ite (= #b1 ((_ extract 29 29) board)) #b0011101 #b0000000)
                  (ite (= #b1 ((_ extract 30 30) board)) #b0011110 #b0000000)
                  (ite (= #b1 ((_ extract 31 31) board)) #b0011111 #b0000000)
                  (ite (= #b1 ((_ extract 32 32) board)) #b0100000 #b0000000)
                  (ite (= #b1 ((_ extract 33 33) board)) #b0100001 #b0000000)
                  (ite (= #b1 ((_ extract 34 34) board)) #b0100010 #b0000000)
                  (ite (= #b1 ((_ extract 35 35) board)) #b0100011 #b0000000)
                  (ite (= #b1 ((_ extract 36 36) board)) #b0100100 #b0000000)
                  (ite (= #b1 ((_ extract 37 37) board)) #b0100101 #b0000000)
                  (ite (= #b1 ((_ extract 38 38) board)) #b0100110 #b0000000)
                  (ite (= #b1 ((_ extract 39 39) board)) #b0100111 #b0000000)
                  (ite (= #b1 ((_ extract 40 40) board)) #b0101000 #b0000000)
                  (ite (= #b1 ((_ extract 41 41) board)) #b0101001 #b0000000)
                  (ite (= #b1 ((_ extract 42 42) board)) #b0101010 #b0000000)
                  (ite (= #b1 ((_ extract 43 43) board)) #b0101011 #b0000000)
                  (ite (= #b1 ((_ extract 44 44) board)) #b0101100 #b0000000)
                  (ite (= #b1 ((_ extract 45 45) board)) #b0101101 #b0000000)
                  (ite (= #b1 ((_ extract 46 46) board)) #b0101110 #b0000000)
                  (ite (= #b1 ((_ extract 47 47) board)) #b0101111 #b0000000)
                  (ite (= #b1 ((_ extract 48 48) board)) #b0110000 #b0000000)
                  (ite (= #b1 ((_ extract 49 49) board)) #b0110001 #b0000000)
                  (ite (= #b1 ((_ extract 50 50) board)) #b0110010 #b0000000)
                  (ite (= #b1 ((_ extract 51 51) board)) #b0110011 #b0000000)
                  (ite (= #b1 ((_ extract 52 52) board)) #b0110100 #b0000000)
                  (ite (= #b1 ((_ extract 53 53) board)) #b0110101 #b0000000)
                  (ite (= #b1 ((_ extract 54 54) board)) #b0110110 #b0000000)
                  (ite (= #b1 ((_ extract 55 55) board)) #b0110111 #b0000000)
                  (ite (= #b1 ((_ extract 56 56) board)) #b0111000 #b0000000)
                  (ite (= #b1 ((_ extract 57 57) board)) #b0111001 #b0000000)
                  (ite (= #b1 ((_ extract 58 58) board)) #b0111010 #b0000000)
                  (ite (= #b1 ((_ extract 59 59) board)) #b0111011 #b0000000)
                  (ite (= #b1 ((_ extract 60 60) board)) #b0111100 #b0000000)
                  (ite (= #b1 ((_ extract 61 61) board)) #b0111101 #b0000000)
                  (ite (= #b1 ((_ extract 62 62) board)) #b0111110 #b0000000)
                  (ite (= #b1 ((_ extract 63 63) board)) #b0111111 #b0000000)
                  prize_index)))
  (bvult a!1 #b1000000)))
(assert (let ((a!1 (bvxor #b0000000
                  (ite (= #b1 ((_ extract 0 0) board)) #b0000000 #b0000000)
                  (ite (= #b1 ((_ extract 1 1) board)) #b0000001 #b0000000)
                  (ite (= #b1 ((_ extract 2 2) board)) #b0000010 #b0000000)
                  (ite (= #b1 ((_ extract 3 3) board)) #b0000011 #b0000000)
                  (ite (= #b1 ((_ extract 4 4) board)) #b0000100 #b0000000)
                  (ite (= #b1 ((_ extract 5 5) board)) #b0000101 #b0000000)
                  (ite (= #b1 ((_ extract 6 6) board)) #b0000110 #b0000000)
                  (ite (= #b1 ((_ extract 7 7) board)) #b0000111 #b0000000)
                  (ite (= #b1 ((_ extract 8 8) board)) #b0001000 #b0000000)
                  (ite (= #b1 ((_ extract 9 9) board)) #b0001001 #b0000000)
                  (ite (= #b1 ((_ extract 10 10) board)) #b0001010 #b0000000)
                  (ite (= #b1 ((_ extract 11 11) board)) #b0001011 #b0000000)
                  (ite (= #b1 ((_ extract 12 12) board)) #b0001100 #b0000000)
                  (ite (= #b1 ((_ extract 13 13) board)) #b0001101 #b0000000)
                  (ite (= #b1 ((_ extract 14 14) board)) #b0001110 #b0000000)
                  (ite (= #b1 ((_ extract 15 15) board)) #b0001111 #b0000000)
                  (ite (= #b1 ((_ extract 16 16) board)) #b0010000 #b0000000)
                  (ite (= #b1 ((_ extract 17 17) board)) #b0010001 #b0000000)
                  (ite (= #b1 ((_ extract 18 18) board)) #b0010010 #b0000000)
                  (ite (= #b1 ((_ extract 19 19) board)) #b0010011 #b0000000)
                  (ite (= #b1 ((_ extract 20 20) board)) #b0010100 #b0000000)
                  (ite (= #b1 ((_ extract 21 21) board)) #b0010101 #b0000000)
                  (ite (= #b1 ((_ extract 22 22) board)) #b0010110 #b0000000)
                  (ite (= #b1 ((_ extract 23 23) board)) #b0010111 #b0000000)
                  (ite (= #b1 ((_ extract 24 24) board)) #b0011000 #b0000000)
                  (ite (= #b1 ((_ extract 25 25) board)) #b0011001 #b0000000)
                  (ite (= #b1 ((_ extract 26 26) board)) #b0011010 #b0000000)
                  (ite (= #b1 ((_ extract 27 27) board)) #b0011011 #b0000000)
                  (ite (= #b1 ((_ extract 28 28) board)) #b0011100 #b0000000)
                  (ite (= #b1 ((_ extract 29 29) board)) #b0011101 #b0000000)
                  (ite (= #b1 ((_ extract 30 30) board)) #b0011110 #b0000000)
                  (ite (= #b1 ((_ extract 31 31) board)) #b0011111 #b0000000)
                  (ite (= #b1 ((_ extract 32 32) board)) #b0100000 #b0000000)
                  (ite (= #b1 ((_ extract 33 33) board)) #b0100001 #b0000000)
                  (ite (= #b1 ((_ extract 34 34) board)) #b0100010 #b0000000)
                  (ite (= #b1 ((_ extract 35 35) board)) #b0100011 #b0000000)
                  (ite (= #b1 ((_ extract 36 36) board)) #b0100100 #b0000000)
                  (ite (= #b1 ((_ extract 37 37) board)) #b0100101 #b0000000)
                  (ite (= #b1 ((_ extract 38 38) board)) #b0100110 #b0000000)
                  (ite (= #b1 ((_ extract 39 39) board)) #b0100111 #b0000000)
                  (ite (= #b1 ((_ extract 40 40) board)) #b0101000 #b0000000)
                  (ite (= #b1 ((_ extract 41 41) board)) #b0101001 #b0000000)
                  (ite (= #b1 ((_ extract 42 42) board)) #b0101010 #b0000000)
                  (ite (= #b1 ((_ extract 43 43) board)) #b0101011 #b0000000)
                  (ite (= #b1 ((_ extract 44 44) board)) #b0101100 #b0000000)
                  (ite (= #b1 ((_ extract 45 45) board)) #b0101101 #b0000000)
                  (ite (= #b1 ((_ extract 46 46) board)) #b0101110 #b0000000)
                  (ite (= #b1 ((_ extract 47 47) board)) #b0101111 #b0000000)
                  (ite (= #b1 ((_ extract 48 48) board)) #b0110000 #b0000000)
                  (ite (= #b1 ((_ extract 49 49) board)) #b0110001 #b0000000)
                  (ite (= #b1 ((_ extract 50 50) board)) #b0110010 #b0000000)
                  (ite (= #b1 ((_ extract 51 51) board)) #b0110011 #b0000000)
                  (ite (= #b1 ((_ extract 52 52) board)) #b0110100 #b0000000)
                  (ite (= #b1 ((_ extract 53 53) board)) #b0110101 #b0000000)
                  (ite (= #b1 ((_ extract 54 54) board)) #b0110110 #b0000000)
                  (ite (= #b1 ((_ extract 55 55) board)) #b0110111 #b0000000)
                  (ite (= #b1 ((_ extract 56 56) board)) #b0111000 #b0000000)
                  (ite (= #b1 ((_ extract 57 57) board)) #b0111001 #b0000000)
                  (ite (= #b1 ((_ extract 58 58) board)) #b0111010 #b0000000)
                  (ite (= #b1 ((_ extract 59 59) board)) #b0111011 #b0000000)
                  (ite (= #b1 ((_ extract 60 60) board)) #b0111100 #b0000000)
                  (ite (= #b1 ((_ extract 61 61) board)) #b0111101 #b0000000)
                  (ite (= #b1 ((_ extract 62 62) board)) #b0111110 #b0000000)
                  (ite (= #b1 ((_ extract 63 63) board)) #b0111111 #b0000000)
                  prize_index)))
(let ((a!2 ((_ extract 0 0)
             (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!3 ((_ extract 1 1)
             (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!4 ((_ extract 2 2)
             (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!5 ((_ extract 3 3)
             (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!6 ((_ extract 4 4)
             (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!7 ((_ extract 5 5)
             (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!8 ((_ extract 6 6)
             (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!9 ((_ extract 7 7)
             (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!10 ((_ extract 8 8)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!11 ((_ extract 9 9)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!12 ((_ extract 10 10)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!13 ((_ extract 11 11)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!14 ((_ extract 12 12)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!15 ((_ extract 13 13)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!16 ((_ extract 14 14)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!17 ((_ extract 15 15)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!18 ((_ extract 16 16)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!19 ((_ extract 17 17)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!20 ((_ extract 18 18)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!21 ((_ extract 19 19)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!22 ((_ extract 20 20)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!23 ((_ extract 21 21)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!24 ((_ extract 22 22)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!25 ((_ extract 23 23)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!26 ((_ extract 24 24)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!27 ((_ extract 25 25)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!28 ((_ extract 26 26)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!29 ((_ extract 27 27)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!30 ((_ extract 28 28)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!31 ((_ extract 29 29)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!32 ((_ extract 30 30)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!33 ((_ extract 31 31)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!34 ((_ extract 32 32)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!35 ((_ extract 33 33)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!36 ((_ extract 34 34)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!37 ((_ extract 35 35)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!38 ((_ extract 36 36)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!39 ((_ extract 37 37)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!40 ((_ extract 38 38)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!41 ((_ extract 39 39)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!42 ((_ extract 40 40)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!43 ((_ extract 41 41)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!44 ((_ extract 42 42)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!45 ((_ extract 43 43)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!46 ((_ extract 44 44)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!47 ((_ extract 45 45)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!48 ((_ extract 46 46)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!49 ((_ extract 47 47)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!50 ((_ extract 48 48)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!51 ((_ extract 49 49)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!52 ((_ extract 50 50)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!53 ((_ extract 51 51)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!54 ((_ extract 52 52)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!55 ((_ extract 53 53)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!56 ((_ extract 54 54)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!57 ((_ extract 55 55)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!58 ((_ extract 56 56)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!59 ((_ extract 57 57)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!60 ((_ extract 58 58)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!61 ((_ extract 59 59)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!62 ((_ extract 60 60)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!63 ((_ extract 61 61)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!64 ((_ extract 62 62)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1)))))
      (a!65 ((_ extract 63 63)
              (bvxor board (bvshl #x0000000000000001 ((_ zero_extend 57) a!1))))))
(let ((a!66 (= (bvxor #b0000000
                      (ite (= #b1 a!2) #b0000000 #b0000000)
                      (ite (= #b1 a!3) #b0000001 #b0000000)
                      (ite (= #b1 a!4) #b0000010 #b0000000)
                      (ite (= #b1 a!5) #b0000011 #b0000000)
                      (ite (= #b1 a!6) #b0000100 #b0000000)
                      (ite (= #b1 a!7) #b0000101 #b0000000)
                      (ite (= #b1 a!8) #b0000110 #b0000000)
                      (ite (= #b1 a!9) #b0000111 #b0000000)
                      (ite (= #b1 a!10) #b0001000 #b0000000)
                      (ite (= #b1 a!11) #b0001001 #b0000000)
                      (ite (= #b1 a!12) #b0001010 #b0000000)
                      (ite (= #b1 a!13) #b0001011 #b0000000)
                      (ite (= #b1 a!14) #b0001100 #b0000000)
                      (ite (= #b1 a!15) #b0001101 #b0000000)
                      (ite (= #b1 a!16) #b0001110 #b0000000)
                      (ite (= #b1 a!17) #b0001111 #b0000000)
                      (ite (= #b1 a!18) #b0010000 #b0000000)
                      (ite (= #b1 a!19) #b0010001 #b0000000)
                      (ite (= #b1 a!20) #b0010010 #b0000000)
                      (ite (= #b1 a!21) #b0010011 #b0000000)
                      (ite (= #b1 a!22) #b0010100 #b0000000)
                      (ite (= #b1 a!23) #b0010101 #b0000000)
                      (ite (= #b1 a!24) #b0010110 #b0000000)
                      (ite (= #b1 a!25) #b0010111 #b0000000)
                      (ite (= #b1 a!26) #b0011000 #b0000000)
                      (ite (= #b1 a!27) #b0011001 #b0000000)
                      (ite (= #b1 a!28) #b0011010 #b0000000)
                      (ite (= #b1 a!29) #b0011011 #b0000000)
                      (ite (= #b1 a!30) #b0011100 #b0000000)
                      (ite (= #b1 a!31) #b0011101 #b0000000)
                      (ite (= #b1 a!32) #b0011110 #b0000000)
                      (ite (= #b1 a!33) #b0011111 #b0000000)
                      (ite (= #b1 a!34) #b0100000 #b0000000)
                      (ite (= #b1 a!35) #b0100001 #b0000000)
                      (ite (= #b1 a!36) #b0100010 #b0000000)
                      (ite (= #b1 a!37) #b0100011 #b0000000)
                      (ite (= #b1 a!38) #b0100100 #b0000000)
                      (ite (= #b1 a!39) #b0100101 #b0000000)
                      (ite (= #b1 a!40) #b0100110 #b0000000)
                      (ite (= #b1 a!41) #b0100111 #b0000000)
                      (ite (= #b1 a!42) #b0101000 #b0000000)
                      (ite (= #b1 a!43) #b0101001 #b0000000)
                      (ite (= #b1 a!44) #b0101010 #b0000000)
                      (ite (= #b1 a!45) #b0101011 #b0000000)
                      (ite (= #b1 a!46) #b0101100 #b0000000)
                      (ite (= #b1 a!47) #b0101101 #b0000000)
                      (ite (= #b1 a!48) #b0101110 #b0000000)
                      (ite (= #b1 a!49) #b0101111 #b0000000)
                      (ite (= #b1 a!50) #b0110000 #b0000000)
                      (ite (= #b1 a!51) #b0110001 #b0000000)
                      (ite (= #b1 a!52) #b0110010 #b0000000)
                      (ite (= #b1 a!53) #b0110011 #b0000000)
                      (ite (= #b1 a!54) #b0110100 #b0000000)
                      (ite (= #b1 a!55) #b0110101 #b0000000)
                      (ite (= #b1 a!56) #b0110110 #b0000000)
                      (ite (= #b1 a!57) #b0110111 #b0000000)
                      (ite (= #b1 a!58) #b0111000 #b0000000)
                      (ite (= #b1 a!59) #b0111001 #b0000000)
                      (ite (= #b1 a!60) #b0111010 #b0000000)
                      (ite (= #b1 a!61) #b0111011 #b0000000)
                      (ite (= #b1 a!62) #b0111100 #b0000000)
                      (ite (= #b1 a!63) #b0111101 #b0000000)
                      (ite (= #b1 a!64) #b0111110 #b0000000)
                      (ite (= #b1 a!65) #b0111111 #b0000000))
               prize_index)))
  (not a!66)))))

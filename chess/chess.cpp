#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using uz = unsigned;

struct Board {
  std::uint64_t bits;
  uz size;
  using BinOp = std::function<uz(uz, uz)>;
  BinOp binop;
  Board(uz size) : size(size), bits(0), binop(std::multiplies<uz>()) {
    assert(size < 8);
  }
  Board(uz size, BinOp binop) : size(size), bits(0), binop(binop) {
    assert(size < 8);
  }
  void walk(std::function<void(uz)> f) {
    for (auto i = 0u; i < pow(2, size * size); i++) {
      bits = i;
      f(i);
    }
  }
  uz sum() const {
    uz t = 0;
    for (auto i = 0u; i < size * size; i++) {
      uz v = (bits & (1 << i)) ? i : 0;
      t = op(t, v);
    }
    return t;
  }
  uz op(uz l, uz r) const { return (l + r) % (size * size); }
  friend std::ostream &operator<<(std::ostream &os, Board &);
};

inline uz operator-(const Board &l, const Board &r) {
  return std::bitset<64>(l.bits ^ r.bits).count();
}

std::ostream &operator<<(std::ostream &os, Board const &b) {
  for (auto i = 0u; i < b.size * b.size; i++) {
    os << ((b.bits & (1 << i)) ? 1 : 0) << " ";
  }
  return os;
}

void check(uz size, Board::BinOp op) {
  Board b(size, op);
  b.walk([=](uz i) {
    std::cout << b;
    std::cout << std::setw(3) << "i=" << i << "->" << std::setw(3) << b.sum()
              << ":\nDistance to:\n";
    Board b2(size, op);
    b2.walk([=](uz i2) {
      if (i2 == i)
        return;
      uz delta = b2 - b;
      if (delta == 1)
        std::cout << "  " << b2 << " (" << b2.sum() << ")"
                  << " is: " << (b2 - b) << "\n";
    });
  });
}

int main(int argc, char **argv) {
  std::vector<std::string> args(argv, argv + argc);
  if (argc < 2)
    return 0;

  unsigned size = std::atoi(args[1].c_str());

  check(size, std::plus<uz>());
  check(size, std::multiplies<uz>());

  return 0;
}

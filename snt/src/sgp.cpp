#include <gecode/driver.hh>
#include <gecode/int.hh>
#include <gecode/set.hh>
#include <gecode/gist.hh>
   
using namespace Gecode;

// 11-11-12
const std::vector<int> GS1_11 =
    {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
    2, 11, 1, 3, 4, 5, 6, 7, 8, 9, 10, 
    3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 
    4, 3, 5, 6, 7, 8, 9, 10, 11, 2, 1, 
    5, 4, 6, 7, 8, 9, 10, 11, 2, 1, 3, 
    6, 5, 7, 8, 9, 10, 11, 2, 1, 3, 4, 
    7, 6, 8, 9, 10, 11, 2, 1, 3, 4, 5, 
    8, 7, 9, 10, 11, 2, 1, 3, 4, 5, 6, 
    9, 8, 10, 11, 2, 1, 3, 4, 5, 6, 7, 
    10, 9, 11, 2, 1, 3, 4, 5, 6, 7, 8, 
    11, 10, 2, 1, 3, 4, 5, 6, 7, 8, 9, 
    };
// 8-8-9
const std::vector<int> GS1_8 =
    {
    1,2,3,4,5,6,7,8,
    2,1,4,3,6,5,8,7,
    3,4,1,2,7,8,5,6,
    4,3,2,1,8,7,6,5,
    5,6,7,8,1,2,3,4,
    6,5,8,7,2,1,4,3,
    7,8,5,6,3,4,1,2,
    8,7,6,5,4,3,2,1,
    };
// 13-13-14
const std::vector<int> GS1_13 =
    {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
    2, 13, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 
    4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2, 1, 
    5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 2, 1, 3, 
    6, 5, 7, 8, 9, 10, 11, 12, 13, 2, 1, 3, 4, 
    7, 6, 8, 9, 10, 11, 12, 13, 2, 1, 3, 4, 5, 
    8, 7, 9, 10, 11, 12, 13, 2, 1, 3, 4, 5, 6, 
    9, 8, 10, 11, 12, 13, 2, 1, 3, 4, 5, 6, 7, 
    10, 9, 11, 12, 13, 2, 1, 3, 4, 5, 6, 7, 8, 
    11, 10, 12, 13, 2, 1, 3, 4, 5, 6, 7, 8, 9, 
    12, 11, 13, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 
    13, 12, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
    };
// 17-17-18
const std::vector<int> GS1_17 =
    {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
    2, 17, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
    3, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 
    4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 1, 
    5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 1, 3, 
    6, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 1, 3, 4, 
    7, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 1, 3, 4, 5, 
    8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 2, 1, 3, 4, 5, 6, 
    9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 2, 1, 3, 4, 5, 6, 7, 
    10, 9, 11, 12, 13, 14, 15, 16, 17, 2, 1, 3, 4, 5, 6, 7, 8, 
    11, 10, 12, 13, 14, 15, 16, 17, 2, 1, 3, 4, 5, 6, 7, 8, 9, 
    12, 11, 13, 14, 15, 16, 17, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 
    13, 12, 14, 15, 16, 17, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
    14, 13, 15, 16, 17, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    15, 14, 16, 17, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
    16, 15, 17, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
    17, 16, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
    };

class GolfOptions : public Options {
    protected:
        Driver::IntOption _w; // Number of weeks
        Driver::IntOption _g; // Number of groups
        Driver::IntOption _s; // Number of players per group
    public:
        // Constructor
        GolfOptions(void) : Options("Golf"), _w("w","number of weeks",9), _g("g","number of groups",8), _s("s","number of players per group",4) {
            add(_w);
            add(_g);
            add(_s);
        }
        // Return number of weeks
        int w(void) const { return _w.value(); }
        // Return number of groups
        int g(void) const { return _g.value(); }
        // Return number of players per group
        int s(void) const { return _s.value(); }
};

/**
 * Schedule a golf tournament. This is problem 010 from csplib.
 */
class Golf : public Script {
    public:
        int g; // Number of groups in a week
        int s; // Number of players in a group
        int w; // Number of weeks

        // The sets representing the groups
        IntVarArray groups;

        // Actual model
        Golf(const GolfOptions& opt) : Script(opt), g(opt.g()), s(opt.s()), w(opt.w()) {
            groups = IntVarArray(*this, g*s*w, 1, g);
            Matrix<IntVarArray> G(groups,g*s,w);
            
            int m = binomialCoefficients(g*s,2);
            Matrix<IntVarArgs> C(IntVarArgs(w * m), m, w);

            TupleSet PT(g);

            // 5-3-7
            if (g==5 && s==3 && w==7) {       
                // initiate constraints 
                constraint1(G);
                constraint2(G);
                constraint3(G);
                constraint4(G, C, m);
                constraint5(G);
                // set brancher
                branch(*this, groups, INT_VAR_MIN_MAX(), INT_VAL_MIN(), nullptr, &printGroups);
            }

            // 5-5-6, 6-6-7, 7-7-8 
            if ((g==5 && s==5 && w==6) || (g==6 && s==6 && w==7) || (g==7 && s==7 && w==8)) {
                // initiate constraints 
                constraint1(G);
                constraint2(G);
                constraint7(G);
                constraint3(G);
                constraint4(G, C, m);
                constraint5(G);
                constraint6(G);
                constraint8(G);
                // set brancher
                branch(*this, groups, INT_VAR_SIZE_MIN(), INT_VAL_MIN(), nullptr, &printGroups);
            }

            // 8-8-9
            if (g==8 && s==8 && w==9) {
                // initiate constraints 
                constraint1(G);
                constraint2(G);
                constraint7(G);
                constraint9(G, GS1_8, PT);
                constraint3(G);
                constraint4(G, C, m);
                constraint5(G);
                constraint6(G);
                constraint10(G, PT);
                // set brancher
                branch(*this, groups, INT_VAR_SIZE_MIN(), INT_VAL_MIN(), nullptr, &printGroups);
            }

            // 11-11-12
            if (g==11 && s==11 && w==12) {
                // initiate constraints 
                constraint1(G);
                constraint2(G);
                constraint7(G);
                constraint9(G, GS1_11, PT);
                constraint11(G);
                constraint3(G);
                constraint4(G, C, m);
                constraint5(G);
                constraint6(G);
                constraint10(G, PT);
                // set brancher
                branch(*this, groups, INT_VAR_SIZE_MIN(), INT_VAL_MIN(), nullptr, &printGroups);
            }

            // 13-13-14
            if (g==13 && s==13 && w==14) {
                // initiate constraints 
                constraint1(G);
                constraint2(G);
                constraint7(G);
                constraint9(G, GS1_13, PT);
                constraint17(G);
                constraint11(G);
                constraint3(G);
                constraint4(G, C, m);
                constraint5(G);
                constraint6(G);
                constraint10(G, PT);
                // set brancher
                branch(*this, groups, INT_VAR_SIZE_MIN(), INT_VAL_MIN(), nullptr, &printGroups);
            }

            // 17-17-18
            if (g==17 && s==17 && w==18) {
                // initiate constraints 
                constraint1(G);
                constraint2(G);
                constraint7(G);
                constraint9(G, GS1_17, PT);
                constraint17(G);
                constraint18(G);
                constraint11(G);
                constraint3(G);
                constraint4(G, C, m);
                constraint5(G);
                constraint6(G);
                constraint10(G, PT);
                // set brancher
                branch(*this, groups, INT_VAR_SIZE_MIN(), INT_VAL_MIN(), nullptr, &printGroups);
            }
        }

        // Print solution
        virtual void print(std::ostream& os) const {
            os << "Tournament plan:\nnote:>\tThe matrix G represents that player j (column index)\n\tis assigned to group G[i][j] in week i (row index)." << std::endl << std::endl;
            Matrix<IntVarArray> schedule(groups,g*s,w);
            for (int j = 0; j < w; j++) {
                os << "Week " << j << ": " << " " << schedule.row(j) << std::endl;
            }
        }

        static void printGroups(const Space& home, 
                                const Brancher& b, 
                                unsigned int a,
                                IntVar s, int i, const int& n,
                                std::ostream& o) {
            const Golf& g = static_cast<const Golf&>(home);
            int col = i % (g.g*g.s);
            int row = i / (g.g*g.s);
            o << "group[" << row  << ", " << col << "] "
            << ((a==0) ? "=" : "=") << " "
            << n << " " << s;
        }

        // Constructor for copying \a s
        Golf(Golf& s) : Script(s), g(s.g), s(s.s), w(s.w) {
            groups.update(*this, s.groups);
        }

        // Copy during cloning
        virtual Space* copy(void) {
            return new Golf(*this);
        }
 
    private:
    
        int binomialCoefficients(int n, int k) {
            if (k == 0 || k == n)
                return 1;
            return binomialCoefficients(n - 1, k - 1) + binomialCoefficients(n - 1, k);
        }

        void constraint1(Matrix<IntVarArray>& G) {
            // eliminate the symmetries among players by fixing the first week
            for (int col = 0; col < g*s; col++) {
                int group = (col / s) + 1;
                G(col,0) = IntVar(*this, group, group);
            }
        }

        void constraint2(Matrix<IntVarArray>& G) {
            // significantly reduce the search space by assigning the first 's' 
            // players to the first 's' groups after the first week
            for (int row = 1; row < w; row++) 
                for (int col = 0; col < s; col++) {
                    int group = col + 1;
                    G(col,row) = IntVar(*this, group, group);
                }
        }

        void constraint3(Matrix<IntVarArray>& G) {
            // each group includes exactly 's' players.
            for (int i = 1; i < w; i++) 
                count(*this, G.row(i), IntSet(s,s), IntArgs::create(g,1,1));
        }

        void constraint4(Matrix<IntVarArray>& G, Matrix<IntVarArgs>& C, int m) {
            // no player meets any other player more than once
            for (int row = 0, c_col = 0; row < w; row++, c_col = 0) 
                for (int col1 = 0; col1 < (g*s); col1++) 
                    for (int col2 = col1+1; col2 < (g*s); col2++) 
                        C(c_col++,row) = expr(*this, G(col1, row) - G(col2, row));

            for (int col = 0; col < m; col++) 
                atmost(*this, C.col(col), 0, 1);
        }

        void constraint5(Matrix<IntVarArray>& G) {
            // players which plays in first week together are pairwise 
            // distinct in other weeks
            for (int row = 1; row < w; row++) {
                for (int pod = s; pod < g*s; pod+=s) {
                    IntVarArgs tmp;
                    for (int pod_col = 0; pod_col < s; pod_col++) 
                        tmp << G(pod + pod_col, row);
                    distinct(*this, tmp);
                }
            }
        }

        void constraint6(Matrix<IntVarArray>& G) {
            // forall GSi submatrices i >= 1 each column distinct
            for (int col = s; col < g*s; col++) {
                IntVarArgs tmp;
                for (int row = 1; row < w; row ++) 
                    tmp << G(col, row);
                distinct(*this, tmp);
            }
        }

        void constraint7(Matrix<IntVarArray>& G) {
            // fix the second row of matrix G
            for (int col = s; col < g*s; col++) {
                int group = (col%s) + 1;
                G(col, 1) = IntVar(*this, group, group);
            }
        }

        void constraint8(Matrix<IntVarArray>& G) {
            // the entries of GS1 are symmetric with respect to
            // the main diagonal. 
            for (int row = 1; row < w; row++) 
                for (int col = s; col < 2*s; col++) 
                    rel(*this, G(col,row), IRT_EQ, G((row+s-1),(col-s+1)));
        }

        void constraint9(Matrix<IntVarArray>& G, const std::vector<int>& init, TupleSet& PT) {
            // use module LatinSquare (make square ...) to get GS1 matrix
            // find matrix in 'square.txt' and copy here
            // note: instance must be 'prime-prime-(prime+1)'
            Matrix<IntArgs> GS1(IntArgs(init), g);
            for (int i = 0; i < g; i++) 
                PT.add(GS1.row(i));
            PT.finalize();

            // fix the GS1 matrix
            for (int row = 1, j=0; row < w; row ++, j++) {
                for (int col = s, i=0; col < 2*s; col++,i++) {
                    G(col, row) = IntVar(*this, GS1(i,j), GS1(i,j));
                }
            }
        }

        void constraint10(Matrix<IntVarArray>& G, TupleSet& PT) {
            //  n limit the row space of the submatrices except GS0 and
            // GS1 to PT (each row of each submatrix can take only values
            // from tuples in PT) 
            for (int row = 1; row < w; row++) {
                for (int pod = 2*s; pod < g*s; pod+=s) {
                    IntVarArgs tmp;
                    for (int pod_col = 0; pod_col < s; pod_col++) 
                        tmp << G(pod + pod_col, row);
                    extensional(*this, tmp, PT);
                }
            } 
        }

        void constraint11(Matrix<IntVarArray>& G) {
            // fix the diagonal of submatrix GS2
            for (int row = 2, i = 0; row < w; row++, i++) {
                G((row+(2*s)-1), row) = IntVar(*this, 1, 1);
            }
        }

        void constraint12(Matrix<IntVarArray>& G) {
            // fix the diagonal of submatrix GS1
            for (int row = 2; row < w; row++) {
                G((row+s-1), row) = IntVar(*this, 1, 1);
            }
        }

        void constraint13(Matrix<IntVarArray>& G) {
            // Eq(x). fixing the second row of GS1 with [2,9,1,3,4,5,6,7,8] when solving 9-9-10
            int values[] = {2,s,1,3,4,5,6,7,8};
            for (int col = s, i=0; col < 2*s; col++,i++) {
                G(col, 2) = IntVar(*this, values[i],values[i]);
            }
        }

        void constraint14(Matrix<IntVarArray>& G) {
            // the main diagonal of the submatrix GS1 is pairwise
            // distinct for 5-5-6 and 7-7-8
            IntVarArgs diagonal;
            for (int row = 1; row < w; row++) 
                diagonal << G((row+s-1), row); 
            distinct(*this, diagonal);
        }

        void constraint15(Matrix<IntVarArray>& G) {
            // players which plays in second week together are pairwise 
            // distinct in other weeks
            for (int row = 2; row < w; row++) {
                for (int group = 0; group < s; group++) {
                    IntVarArgs tmp;
                    for (int col = 0; col < g*s; col++) {
                        if (col % s == group) { tmp << G(col, row);}
                    }
                    distinct(*this, tmp);
                }
            }
        }

        void constraint16(Matrix<IntVarArray>& G) {
            // forall GSi submatrices i >= 3 each diagonal distinct
            for (int pod = 2*s; pod < g*s; pod+=s) {
                IntVarArgs diagonal;
                for (int row = 1; row < w; row ++) 
                    diagonal << G((pod+row-1),row);
                distinct(*this, diagonal);
            }
        }

        void constraint17(Matrix<IntVarArray>& G) {
            // fix the third row in p-p-(p+1) matrices
            for (int col = 2*s; col < g*s; col+= s) {
                int val = col / s + 1;
                G(col, 2) = IntVar(*this, val,val);
            }
        }

        void constraint18(Matrix<IntVarArray>& G) {
            // fix the matrix GS3 for 17-17-18
            int val[] = {17, 15, 13, 11, 9, 7, 5, 3, 2, 16, 14, 12, 10, 8, 6}; 
            for (int row = 3, i = 0; row < w; row++, i++) {
                G(3*s, row) = IntVar(*this, val[i],val[i]);
            }
        }
};

int main(int argc, char* argv[]) {
    GolfOptions opt;
    opt.ipl(IPL_BND);
    opt.solutions(1);
    opt.parse(argc,argv);

    Script::run<Golf,DFS,GolfOptions>(opt);
    return 0;
}
